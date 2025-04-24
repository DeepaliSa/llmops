# score.py

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import base64, io, os

def init():
    global pipe
    model_id = "stabilityai/stable-diffusion"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        use_auth_token=os.getenv("HUGGINGFACE_TOKEN")
    )
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

def run(raw_data):
    prompt = raw_data.get("prompt", "a fantasy landscape")
    image = pipe(prompt).images[0]

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return {"image_base64": encoded}
