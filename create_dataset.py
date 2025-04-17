import subprocess
import os
from pydub import AudioSegment
import webvtt
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import csv
import os
import sys

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

def chunk_audio(audio_path, video_title):
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
    totalLength = len(audio)

    # Define length of time for each chunk in seconds
    chunk_length = 10 * 1000  # seconds in ms

    # Get the total amount of chunks needed to divide the audio file
    num_chunks = totalLength // chunk_length + int(totalLength % chunk_length != 0)

    # display expected number of chunks
    print(f'Expected number of chunks: {num_chunks}')

    # Extract the needed number of audio chunks from the audio file
    print(f'Chunks saving into {wavs_dir}')
    for i in range(num_chunks):
        # Get the start and end time for the chunk
        start = i * chunk_length
        end = min((i + 1) * chunk_length, totalLength)

        # Extract audio and output labeled chunk into ouput dir as a wav file
        chunk = audio[start:end]
        chunk.export(os.path.join(wavs_dir, f"chunk_{i:04}.wav"), format="wav")

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

def main():
    # get URL
    if len(sys.argv) < 2:
        print("Usage: python your_script.py <YouTube_URL>")
        sys.exit(1)

    url = sys.argv[1]

    print('Downloading YouTube video...')
    yt_audio_path, title = yt_download(url)
    print(f'YouTube video: {title} downloaded.')

    print('Chunking audio...')
    dataset_dir, wavs_dir = chunk_audio(yt_audio_path, title)
    print('Audio chunking complete.')

    print('Transcribing wavs...')
    csv_path = transcribe_wavs(dataset_dir, wavs_dir)
    print("Transcriptions written to:", csv_path)

if __name__ == "__main__":
    main()