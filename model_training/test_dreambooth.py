from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("./trained_model", torch_dtype=torch.float16, use_safetensors=True).to("cuda")
image = pipeline("picture of (sks dog) in buenos aires obelisk", num_inference_steps=50, guidance_scale=7.5).images[0]
image.save("./data/output/dog-bucket.png")