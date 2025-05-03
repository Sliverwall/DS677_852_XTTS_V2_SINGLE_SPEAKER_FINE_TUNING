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

def chunk_audio(audio_path, video_title, lower_bound, upper_bound):
    # Create dataset folder
    dataset_dir = f"./datasets/{video_title}"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    wavs_dir = os.path.join(dataset_dir, 'wavs')
    if not os.path.exists(wavs_dir):
        os.makedirs(wavs_dir)
    
    # Load audio
    audio = AudioSegment.from_wav(audio_path)  

    # Get the total length of time, in ms, for the audio file
    total_length = len(audio)

    # Define mean & std dev
    mean = (upper_bound + lower_bound)/2
    sd = 1

    # Standardize lower & upper bounds
    std_lower = (lower_bound - mean)/sd
    std_upper = (upper_bound - mean)/sd

    # Gen random variable with params
    X = truncnorm(
        a = std_lower,
        b = std_upper,
        loc = mean,
        scale = sd
    )

    # Gen samples
    data = X.rvs(1000)

    # Round data to whole seconds
    data = np.round(data, decimals=3)

    # convert to ms
    data_ms = data * 1000

    # Define generation limit in chunks
    gen_limit = total_length - upper_bound
    
    chunks = []
    gen_chunks = 0
    num_chunks = 0

    # Extract the needed number of audio chunks from the audio file
    print(f'Saving chunks into {wavs_dir}')
    while gen_chunks < gen_limit:
        # Sample from gen normal distrubtion
        chunk_len = np.random.choice(data_ms)
        
        # Get the start and end time for the chunk
        start = gen_chunks
        end = min(start + chunk_len, total_length)
        
        # Extract audio and output labeled chunk into ouput dir as a wav file
        chunk = audio[start:end]
        chunk.export(os.path.join(wavs_dir, f"chunk_{num_chunks:04}.wav"), format="wav")
        
        # Add chunk size to tracker
        chunks.append(end-start)
        
        # Add chunk length to generatedChunk counter
        gen_chunks += chunk_len

        # add to iterate value
        num_chunks += 1

    return dataset_dir, wavs_dir

def transcribe_wavs(dataset_dir, wavs_dir):
    # Set device and torch data type
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f'Using device: {device}')

    # Define chunk size used in seconds
    chunk_size = 10

    # Model identifier
    model_id = "openai/whisper-large-v3"

    # Load the model and move it to the selected device
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, 
        torch_dtype=torch_dtype, 
        low_cpu_mem_usage=False, 
        use_safetensors=True
    )
    model.to(device)

    # Load the processor
    processor = AutoProcessor.from_pretrained(model_id, language='en')

    # Add a new special pad token (string) to the tokenizer
    if processor.tokenizer.pad_token_id == processor.tokenizer.eos_token_id:
        processor.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        processor.tokenizer.pad_token_id = processor.tokenizer.convert_tokens_to_ids('[PAD]')

    # Create the speech recognition pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=chunk_size,
        batch_size=32, 
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )

    # Define a path to an output CSV to save transcriptions
    csv_path = os.path.join(dataset_dir, "metadata.csv")
    file_list = sorted(os.listdir(wavs_dir))

    # Create full audio paths
    audio_paths = [os.path.join(wavs_dir, file) for file in file_list]

    # Run the pipeline on all files in batch
    results = pipe(audio_paths)

    # Format results in LJ Speech style
    samples = [
        (os.path.splitext(os.path.basename(audio_paths[i]))[0], 
         results[i]["text"], 
         results[i]["text"])
        for i in range(len(results))
    ]

    # Write the samples list to output csv
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        csv_writer = csv.writer(f, delimiter='|')
        for entry in samples:
            csv_writer.writerow(entry)

    return csv_path

def clean_transcription(wavs_dir, csv_path):
    MAX_CHAR_LEN = 250
    Z_SCORE_MAX = 2

    # Read metadata
    cols = ["file_name", "text", "normalized_text"]
    df = pd.read_csv(csv_path, sep="|", header=None, names=cols)
    df["text_len"] = df["text"].str.len()

    # Filter entries exceeding max char length
    df = df[df["text_len"] <= MAX_CHAR_LEN]

    # Read wav file durations
    file_list = os.listdir(wavs_dir)
    file_sizes = []

    for file in file_list:
        file_name = os.path.splitext(file)[0]
        audio_path = os.path.join(wavs_dir, file)
        audio_ms = len(AudioSegment.from_wav(audio_path))
        file_sizes.append([file_name, audio_ms])

    pairing_cols = ["file_name", "length_ms"]
    pairing_df = pd.DataFrame(file_sizes, columns=pairing_cols)

    # Merge metadata with audio durations
    stats_df = pd.merge(df, pairing_df, on="file_name", how="left")
    stats_df["ms_per_char"] = stats_df["length_ms"] / stats_df["text_len"]

    # Z-score for outlier detection
    stats_df["z_score"] = zscore(stats_df["ms_per_char"])
    stats_df["outlier"] = stats_df["z_score"].abs() > Z_SCORE_MAX

    # Remove outliers
    stats_df = stats_df[stats_df["outlier"] == False]

    # Filter original metadata to match cleaned set
    output_df = pd.merge(df, stats_df[["file_name"]], on="file_name", how="inner")

    # Overwrite the original metadata file
    output_df.to_csv(csv_path, sep="|", index=False, header=False)

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

    lower_bound, upper_bound = 3, 10
    print('Chunking audio...')
    dataset_dir, wavs_dir = chunk_audio(audio_path, title, lower_bound, upper_bound)
    print('Audio chunking complete.')

    print('Transcribing wavs...')
    csv_path = transcribe_wavs(dataset_dir, wavs_dir)
    print(f"Transcriptions written to: {csv_path}")

if __name__ == "__main__":
    main()