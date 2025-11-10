import torch
import torchaudio
import os
from datetime import datetime
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
from huggingface_hub import login
from dotenv import load_dotenv
import scipy.io.wavfile

# HUGGINGFACE LOGIN
load_dotenv()
login(token=os.getenv("HF_TOKEN"))

# Make sure I'm using GPU lol
device = "cuda" if torch.cuda.is_available() else "cpu"

# Download model (only once when module is imported)
model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]

model = model.to(device)


def generate_audio(prompt: str, duration: int = 15, filename: str = None) -> str:
    """
    Generate audio from a text prompt using Stable Audio Open model.
    
    Args:
        prompt: Text prompt describing the audio to generate
        duration: Duration of the audio in seconds (default: 15)
        filename: Optional custom filename. If None, generates timestamp-based filename.
    
    Returns:
        str: The filename of the generated audio file (relative to app/static/)
    """
    # Set up text and timing conditioning
    conditioning = [{
        "prompt": prompt,
        "seconds_start": 0,
        "seconds_total": duration
    }]

    # Generate stereo audio
    output = generate_diffusion_cond(
        model,
        steps=100,
        cfg_scale=7,
        conditioning=conditioning,
        sample_size=sample_size,
        sigma_min=0.3,
        sigma_max=500,
        sampler_type="dpmpp-3m-sde",
        device=device,
        seed=42  # Use a specific seed to avoid the random seed generation issue
    )

    # Rearrange audio batch to a single sequence
    output = rearrange(output, "b d n -> d (b n)")

    # Peak normalize, clip, convert to int16, and save to file
    output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
    
    # Create app/static directory if it doesn't exist
    static_dir = os.path.join("app", "static")
    os.makedirs(static_dir, exist_ok=True)
    
    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"audio_{timestamp}.wav"
    elif not filename.endswith(".wav"):
        filename = f"{filename}.wav"
    
    # Save to app/static folder using scipy to avoid torchcodec dependency
    save_path = os.path.join(static_dir, filename)
    # Convert to numpy and transpose: scipy expects (samples, channels) format
    output_numpy = output.numpy().T  # Transpose from (channels, samples) to (samples, channels)
    scipy.io.wavfile.write(save_path, sample_rate, output_numpy)
    
    return filename
