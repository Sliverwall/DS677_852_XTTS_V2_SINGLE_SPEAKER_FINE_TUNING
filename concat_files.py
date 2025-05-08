from pydub import AudioSegment
import os

# Set the path to the folder containing WAV files
folder_path = r"C:\Users\caama\Documents\School\NJIT\DS677\Project\script_audio"
output_path = "./concatenated_output.wav"

# Initialize an empty AudioSegment
combined = AudioSegment.empty()

# Get all wav files
wav_files = []
for f in os.listdir(folder_path):
    name, ext = os.path.splitext(f)
    if ext.lower() == ".wav" and name.isdigit():
        wav_files.append((int(name), f))

# Sort by numeric value
wav_files.sort()

# Loop through all .wav files and concatenate
for _, filename in wav_files:
    file_path = os.path.join(folder_path, filename)
    audio = AudioSegment.from_wav(file_path)
    combined += audio

# Export the combined audio
combined.export(output_path, format="wav")
print(f"Combined audio saved to: {output_path}")
