from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import torch
import torchaudio
import os
import re
from typing import Optional

def gen_audio(text: str,
            checkpoint_dir: str,
            reference_wav: str,
            output_path: str,
            vocab_path: Optional[str]="../XTTS-files/vocab.json",
            split_sentences: bool=True,
            temperature: float=0.7,
) -> str:
    # Load device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load config file
    cfg = XttsConfig()
    cfg.load_json(os.path.join(checkpoint_dir, "config.json"))

    # Init model using the config
    model = Xtts.init_from_config(cfg)

    # Load from checkpoint - model gets loaded in using the base learned model/speaker embeedings
    model.load_checkpoint(
        cfg,
        checkpoint_dir=checkpoint_dir,
        vocab_path=vocab_path,
        eval=True,
        strict=True,
        use_deepspeed=False,
    )

    # Set to eval
    model.to(device).eval()
    
    # Get conditioning latents
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=[reference_wav],
        gpt_cond_len=cfg.gpt_cond_len,
        gpt_cond_chunk_len=cfg.gpt_cond_chunk_len,
        max_ref_length=cfg.max_ref_len,
    )
    
    if split_sentences:
        # Break text into distinct sentences
        sentences = re.split(r'(?<=[.!?]) +', text.strip())
    else:
        sentences = [text]

    segments = []

    # Inference one sentence at a time
    for sentence in sentences:
        print(f"Generating audio for: {sentence}")

        out = model.inference(
            text=sentence,
            language="en",
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=temperature,
            speed=0.95,
            length_penalty=cfg.length_penalty,
            repetition_penalty=cfg.repetition_penalty,
            top_k=cfg.top_k,
            top_p=cfg.top_p,
        )
        
        # Create wav tensor then add to segements list
        wav_tensor = torch.tensor(out["wav"]).unsqueeze(0)  # shape: (1, samples)
        segments.append(wav_tensor)

    # Convert the output in wav format, set to tensor for torchaudio
    # Concatenate all wav tensors along the time axis (dim=1)
    finalAudio = torch.cat(segments, dim=1)
    
    torchaudio.save(output_path, finalAudio, sample_rate=cfg.audio.output_sample_rate)
    
    print(f"Output saved to {output_path}")
    
    # Return output path
    return output_path