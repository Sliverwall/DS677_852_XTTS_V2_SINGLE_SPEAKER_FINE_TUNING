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

def chunk_sentences(audio_path, title, min_sec=MIN_SEC, max_sec=MAX_SEC, compute_type=COMPUTE_TYPE, batch_size=BATCH_SIZE, whisper_model=WHISPER_MODEL):
    # Create dataset folder
    dataset_dir = f"./datasets/{title}"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    wavs_dir = os.path.join(dataset_dir, 'wavs')
    if not os.path.exists(wavs_dir):
        os.makedirs(wavs_dir)

    # Set device and torch data type
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using device: {device}')

    print("Transcribing and aligning with WhisperX.")
    model = whisperx.load_model(whisper_model, device=device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=16, language="en")
    model_a, metadata = whisperx.load_align_model(language_code="en", device=device)
    aligned = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    word_segments = aligned["word_segments"]
    full_text = " ".join([w["word"] for w in word_segments])
    sentences = sent_tokenize(full_text)

    sentence_segments = []
    word_index = 0
    for sent in sentences:
        num_words = len(sent.split())
        if word_index + num_words > len(word_segments):
            break
        start = word_segments[word_index]["start"]
        end = word_segments[word_index + num_words - 1]["end"]
        sentence_segments.append({"text": sent, "start": start, "end": end})
        word_index += num_words

    # Merge segments if under threshold
    merged = []
    buffer, buffer_duration = [], 0.0
    for s in sentence_segments:
        dur = s["end"] - s["start"]
        if min_sec <= dur <= max_sec:
            merged.append(s)
        else:
            buffer.append(s)
            buffer_duration += dur
            if buffer_duration >= min_sec:
                merged.append({
                    "text": " ".join(b["text"] for b in buffer),
                    "start": buffer[0]["start"],
                    "end": buffer[-1]["end"]
                })
                buffer = []
                buffer_duration = 0.0

    # Export Chunks and Transcriptions
    audio = AudioSegment.from_file(audio_path)
    metadata_path = os.path.join(dataset_dir, "metadata.csv")
    with open(metadata_path, "w", encoding="utf-8-sig") as f:
        for i, seg in enumerate(merged):
            start_ms = int(seg["start"] * 1000)
            end_ms = int(seg["end"] * 1000)
            chunk = audio[start_ms:end_ms]
            fname = f"chunk_{i:04}.wav"
            chunk.export(os.path.join(wavs_dir, fname), format="wav")
            f.write(f"{os.path.splitext(fname)[0]}|{seg['text']}|{seg['text']}\n")

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