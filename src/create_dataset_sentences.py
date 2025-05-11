import subprocess
import os
import argparse
from pydub import AudioSegment
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import csv
import sys
import re
from scipy.stats import truncnorm, zscore
import numpy as np
import pandas as pd

# Faster inference with TF32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Sentence level transcription
import whisperx
import nltk

NLTK_DIR = "C:/ProgramData/miniconda3/envs/DS677_Project/Lib/nltk_data"

nltk.download('punkt', download_dir=NLTK_DIR)
nltk.download('punkt_tab', download_dir=NLTK_DIR)
nltk.download('wordnet', download_dir=NLTK_DIR)
nltk.download('omw-1.4', download_dir=NLTK_DIR)
os.environ["NLTK_DATA"] = NLTK_DIR
from nltk.tokenize import sent_tokenize

# Define upper & lower bounds for audio in seconds
MIN_SEC, MAX_SEC = 3, 11.6

# Define WhisperX parameters
BATCH_SIZE = 16
COMPUTE_TYPE = "float16"
WHISPER_MODEL = "large-v3" # Options: tiny, base, small, medium, large-v2, large-v3

def yt_download(url):
    """
    Downloads a wav file from given YouTube url.
    """
    # Create audio folder
    audio_dir = f"./audio/"
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)

    # Get video title (used later to create dir)
    result = subprocess.run(
        ["yt-dlp", "--get-title", url],
        capture_output=True, text=True
    )
    title = result.stdout.strip()

    # Clean title by removing special characters
    title = re.sub(r'[^a-zA-Z0-9_\- ]', '', title)

    yt_audio_path = os.path.join(audio_dir, f"{title}.wav")

    subprocess.run([
        "yt-dlp",
        "-f", "bestaudio",
        "--extract-audio",
        "--audio-format", "wav", # convert to wav as this is XTTS prefered format
        "-o", yt_audio_path,
        url
    ])

    return yt_audio_path, title

def load_and_concat(file_paths, title, normalize_dbfs=-20.0):
    # Create audio folder
    audio_dir = f"./audio/"
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)
    
    output_path = os.path.join("audio", f"{title}.wav")

    segments = [AudioSegment.from_file(path) for path in file_paths]
    combined = sum(segments) if len(segments) > 1 else segments[0]

    # Normalize loudness
    print(f"Combined loudness before normalization: {combined.dBFS:.2f} dBFS")
    change_dBFS = normalize_dbfs - combined.dBFS
    combined = combined.apply_gain(change_dBFS)
    print(f"Normalized loudness to {normalize_dbfs} dBFS")

    combined.export(output_path, format="wav")

    return output_path

def resample_audio(audio_path):
    audio = AudioSegment.from_file(audio_path)

    frame_rate = audio.frame_rate
    channels = audio.channels
    sample_width_bits = audio.sample_width * 8

    print(f"Original audio: {frame_rate} Hz, {channels} channel(s), {sample_width_bits}-bit")

    if frame_rate != 22050 or channels != 1 or sample_width_bits != 16:
        print("Resampling to 22050 Hz, mono, 16-bit PCM.")
        resampled = audio.set_frame_rate(22050).set_channels(1).set_sample_width(2)
        resampled.export(audio_path, format="wav")
        print("Resampling complete and file overwritten.")
    else:
        print("Audio already meets the required format.")

    return audio_path

def chunk_sentences(
    audio_path,
    title,
    min_sec: float = MIN_SEC,
    max_sec: float = MAX_SEC,
    char_limit: int = 250,
    compute_type: str = COMPUTE_TYPE,
    batch_size: int = BATCH_SIZE,
    whisper_model: str = WHISPER_MODEL,
):
    dataset_dir = os.path.join("./datasets", title)
    wavs_dir     = os.path.join(dataset_dir, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # WhisperX transcription + alignment
    model      = whisperx.load_model(whisper_model, device=device, compute_type=compute_type)
    audio_arr  = whisperx.load_audio(audio_path)
    base       = model.transcribe(audio_arr, batch_size=batch_size, language="en")
    align_mdl, meta = whisperx.load_align_model("en", device)
    aligned    = whisperx.align(base["segments"], align_mdl, meta, audio_arr,
                                device, return_char_alignments=False)

    word_segs  = aligned["word_segments"]
    full_text  = " ".join(w["word"] for w in word_segs)
    sentences  = sent_tokenize(full_text)

    # map sentences -> (start, end)
    sent_segs, idx = [], 0
    for sent in sentences:
        n = len(sent.split())
        if idx + n > len(word_segs):
            break
        sent_segs.append({
            "text":  sent,
            "start": word_segs[idx]["start"],
            "end":   word_segs[idx + n - 1]["end"],
        })
        idx += n

    # Build chunks respecting all limits
    chunks, buf, dur, txt = [], [], 0.0, ""
    for seg in sent_segs:
        seg_dur  = seg["end"] - seg["start"]
        seg_text = seg["text"]
        # Case 1: sentence is valid and fits limits
        if min_sec <= seg_dur <= max_sec and len(seg_text) <= char_limit:
            if buf:
                chunks.append(buf)
                buf, dur, txt = [], 0.0, ""
            chunks.append([seg])
            continue

        # Case 2: add sentence to buffer
        if dur + seg_dur > max_sec or len(txt) + 1 + len(seg_text) > char_limit:
            if dur >= min_sec:
                chunks.append(buf)
                buf, dur, txt = [], 0.0, ""
            else:
                # buffer too short
                pass

        buf.append(seg)
        dur += seg_dur
        txt += (" " if txt else "") + seg_text

    if buf:
        chunks.append(buf)

    # Write wavs + metadata
    audio               = AudioSegment.from_file(audio_path)
    metadata_path       = os.path.join(dataset_dir, "metadata.csv")
    with open(metadata_path, "w", encoding="utf‑8‑sig") as meta_f:
        for i, chunk in enumerate(chunks):
            start  = int(chunk[0]["start"] * 1000)
            end    = int(chunk[-1]["end"] * 1000)
            if end <= start:
                continue
            piece  = audio[start:end]
            if len(piece) < min_sec*1000 or len(piece) > max_sec*1000:
                continue # double‑check duration
            fname  = f"chunk_{i:04}.wav"
            piece.export(os.path.join(wavs_dir, fname), format="wav")

            text   = " ".join(c["text"] for c in chunk)
            if len(text) > char_limit: # final text safety check
                # Truncate at nearest word before limit
                text = text[:char_limit].rsplit(" ", 1)[0] + " …"

            # Write LJSpeech‑style row: <id>|<text>|<text>
            meta_f.write(f"{os.path.splitext(fname)[0]}|{text}|{text}\n")

    return dataset_dir, wavs_dir

def main():
    parser = argparse.ArgumentParser(description="Create dataset from YouTube or audio files.")
    group = parser.add_mutually_exclusive_group(required=True) # User must specify one or the other
    group.add_argument('--url', type=str, help='YouTube URL to download audio from')
    group.add_argument('--files', nargs='+', help='List of audio files to process directly') # nargs='+': one or more values
    parser.add_argument('--title', type=str, help='Title prefix when using --files')

    args = parser.parse_args()

    # Post-parse validation
    if args.url and args.title:
        print("Error: --title can only be used with --files.")
        sys.exit(1)

    if args.url:
        print('Downloading YouTube video...')
        audio_path, title = yt_download(args.url)
        print(f'YouTube video: {title} downloaded.')
    else:
        title = args.title if args.title else "local_audio"
        print('Loading and concatenating input files.')
        audio_path = load_and_concat(args.files, title)
        print(f"{len(args.files)} file(s) loaded and concatenated.")
        print(f"Audio written to: {audio_path}")

    print('Resampling audio to 24kHz.')
    audio_path = resample_audio(audio_path)
    print('Audio resampling complete.')

    print("Chunking by sentence using WhisperX.")
    dataset_dir, wavs_dir = chunk_sentences(audio_path, title)
    print(f"Chunks and metadata saved in {dataset_dir}")

if __name__ == "__main__":
    main()