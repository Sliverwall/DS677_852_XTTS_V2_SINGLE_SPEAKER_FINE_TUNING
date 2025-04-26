import shutil
from pathlib import Path
from typing import List
from io import BytesIO

from pydub import AudioSegment
import streamlit as st

# Define directories
RECORDINGS_DIR = Path("./recordings")
UPLOADS_DIR = Path("./uploads")
GEN_DIR = Path("./generated_audio")

# Define min & max duration for uploaded/recorded audio
MIN_DURATION_SEC = 3
MAX_DURATION_SEC = 10

def create_dir():
    """Ensure that the necessary directories exist."""
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    GEN_DIR.mkdir(parents=True, exist_ok=True)

def ffmpeg_check() -> bool:
    """Check if FFmpeg is installed."""
    return shutil.which("ffmpeg") is not None

def resample_wav(audio_file, output_path: Path, sample_rate: int=16000):
    """Convert an audio file to WAV format with the specified sample rate.
    
    Args:
        audio_file:
            BytesIO object of the uploaded audio.
        output_path:
            Path to save the converted WAV file.
        sample_rate:
            Desired sample rate in kHz.
    """
    try:
        # Read audio
        if isinstance(audio_file, BytesIO):
            audio = AudioSegment.from_file(audio_file)
        else:
            audio = AudioSegment.from_file(BytesIO(audio_file.read()))

        # Resample & save as wav
        audio = audio.set_frame_rate(sample_rate).set_channels(1)
        audio.export(output_path, format="wav")
    except Exception as e:
        st.error(f"Error converting audio: {e}")

def valid_audio(audio_file: AudioSegment, min_sec: int, max_sec: int) -> bool:
    """Validate audio duration in seconds.
        Args:
            audio_file:
                BytesIO object of the uploaded audio.
            min_sec:
                Lower bound in seconds.
            sample_rate:
                Upper bound in seconds.

        Returns:
            Boolean representing if the audio is valid.
    """
    
    try:
        # Read audio
        if isinstance(audio_file, BytesIO):
            audio = AudioSegment.from_file(audio_file)
        else:
            audio = AudioSegment.from_file(BytesIO(audio_file.read()))

        # Validate duration
        duration_sec = len(audio) / 1000.0
        return min_sec <= duration_sec <= max_sec
    except Exception as e:
        st.error(f"Error validating audio: {e}")
        return False

def list_audio_files(audio_dir: Path) -> List[str]:
    """List all wav audio files in the specified directory.
    
    Args:
        audio_dir: Directory to search for audio files.
    
    Returns:
        List of filenames without extension.
    """
    return [f.stem for f in audio_dir.glob("*.wav")]

def get_example_audio() -> dict:
    """Retrieve lists of default models and user-provided audio samples.
    
    Returns:
        Dictionary with form {'Audio Name': 'Checkpoint Path'}
    """
    audio_mapping = {
        "Sherlock Holmes": r"C:/Users/caama/Documents/School/NJIT/DS677/Project/run/training/Sherlock-Holmes-2-epochs-April-25-2025_03+08PM-0000000",
        "Tom Hanks": "FINETUNED MODEL DIR" # CHANGE TO DIRECTORIES
    }

    # Uploaded and recorded examples (map to default model)
    recordings = list_audio_files(RECORDINGS_DIR)
    uploads = list_audio_files(UPLOADS_DIR)
    user_samples = recordings + uploads

    for sample in user_samples:
        # Map if not in dict
        if sample not in audio_mapping:
            audio_mapping[sample] = "./XTTS-files/"

    return audio_mapping
    