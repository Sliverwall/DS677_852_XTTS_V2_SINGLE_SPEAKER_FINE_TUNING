import streamlit as st
from pathlib import Path

from audio_utils import (
    create_dir,
    ffmpeg_check,
    resample_wav,
    valid_audio,
    get_example_audio,
    RECORDINGS_DIR,
    UPLOADS_DIR,
    GEN_DIR,
    MIN_DURATION_SEC,
    MAX_DURATION_SEC
)
from inference import gen_audio

# Ensure necessary dir are created
create_dir()

# Check FFmpeg install
if not ffmpeg_check():
    st.warning("FFmpeg is not installed or not found in PATH. Audio conversion may fail.")

# App Title
st.title("DS677 XTTS-v2 Demo")

# ---------- Audio Recording or Upload ---------- 
st.header("1. Record / Upload Audio")

record_option = st.radio("Choose input method:", ("Record Audio", "Upload Audio"))

recorded_audio = None
uploaded_audio = None

# Audio recording option
if record_option == "Record Audio":
    st.info("Record your voice using the microphone.")
    recorded_audio = st.audio_input("Record Your Voice")

    if recorded_audio:
        # Verify audio len
        if valid_audio(recorded_audio, MIN_DURATION_SEC, MAX_DURATION_SEC, audio_format="wav"):
            with st.form("save_recording_form"):
                filename_input = st.text_input("Enter a name for the recording:").strip() # Name to save recording
                save_recording = st.form_submit_button("Save Recording")
                if save_recording:
                    if not filename_input:
                        st.warning("You must enter a filename to save the recording.")
                    else:
                        output_path = RECORDINGS_DIR / f"{filename_input}.wav"
                        # Resample & save as wav
                        resample_wav(recorded_audio, output_path, audio_format="wav")
                        st.success(f"Recording saved as '{filename_input}.wav'.")
        else:
            st.error(f"Recording must be between {MIN_DURATION_SEC}-{MAX_DURATION_SEC} seconds.")
# Audio upload option
else:
    uploaded_audio = st.file_uploader("Upload an Audio File (.wav or .mp3)", type=["wav", "mp3"])
    if uploaded_audio:
        file_name = Path(uploaded_audio.name).stem
        file_ext = Path(uploaded_audio.name).suffix.lower()
        # Verify audio len
        if file_ext not in [".wav", ".mp3"]:
            st.error("Unsupported file type. Only .wav and .mp3 are allowed.")
        else:
            audio_format = "wav" if file_ext == ".wav" else "mp3"
            if valid_audio(uploaded_audio, MIN_DURATION_SEC, MAX_DURATION_SEC, audio_format):
                output_path = UPLOADS_DIR / f"{file_name}.wav"
                # Resample & save as wav
                resample_wav(uploaded_audio, output_path, audio_format=audio_format)
                st.success(f"Upload saved as '{file_name}.wav'.")
            else:
                st.error(f"Upload must be between {MIN_DURATION_SEC}-{MAX_DURATION_SEC} seconds.")
        

# ---------- Selecting Example Audio ---------- 
st.header("2. Select Audio")

# Retrieve lists of default models and user-provided audio samples
example_audio_mapping = get_example_audio()
example_audio_names = list(example_audio_mapping.keys())

selected_audio_name = st.selectbox("Select an audio option:", options=example_audio_names)
selected_model = example_audio_mapping[selected_audio_name]

# ---------- Text Input ---------- 
st.header("3. Enter Text")

text_prompt = st.text_area("Enter text to synthesize:", max_chars=500)

# ---------- Generate Audio Button ---------- 
st.header("4. Generate Synthesized Audio")

if "generated_audio_path" not in st.session_state:
    st.session_state["generated_audio_path"] = None

generate_button = st.button("Generate Audio")

if generate_button:
    if not text_prompt.strip():
        st.error("Please enter text before generating.")
    else:
        # Determine the reference audio file path
        if selected_audio_name == "Benedict Cumberbatch":
            reference_audio_path = "C:/Users/caama/Documents/School/NJIT/DS677/Project/datasets/Sherlock Holmes Stories  Read by Benedict Cumberbatch/wavs/chunk_0220.wav"
        elif selected_audio_name == "Tom Hanks":
            reference_audio_path = "C:/Users/caama/Documents/School/NJIT/DS677/Project/models/Tom Hanks/chunk_0009.wav"
        else:
            # Look for the saved user recordings/uploads
            if (RECORDINGS_DIR / f"{selected_audio_name}.wav").exists():
                reference_audio_path = str(RECORDINGS_DIR / f"{selected_audio_name}.wav")
            elif (UPLOADS_DIR / f"{selected_audio_name}.wav").exists():
                reference_audio_path = str(UPLOADS_DIR / f"{selected_audio_name}.wav")
            else:
                st.error("Reference audio file not found.")
                reference_audio_path = None

        if reference_audio_path:
            # Output path
            generated_audio_path = f"{GEN_DIR}/synthesized_{selected_audio_name.replace(' ', '_')}.wav"

            # Inference call
            out = gen_audio(
                text=text_prompt,
                checkpoint_dir=selected_model,
                reference_wav=reference_audio_path,
                output_path=generated_audio_path,
                split_sentences=True
            )

            st.session_state["generated_audio_path"] = generated_audio_path
            st.success("Audio generated successfully!")

# ---------- Playback and Download after Generation ---------- 
if st.session_state["generated_audio_path"] and Path(st.session_state["generated_audio_path"]).exists():
    # First open for playback
    audio_bytes = open(st.session_state["generated_audio_path"], "rb").read()
    st.audio(audio_bytes, format="audio/wav")

    # Open separately for download
    with open(st.session_state["generated_audio_path"], "rb") as f:
        st.download_button(
            label="Download Synthesized Audio",
            data=f,
            file_name=Path(st.session_state["generated_audio_path"]).name,
            mime="audio/wav"
        )