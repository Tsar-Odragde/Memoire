from diffusers import DiffusionPipeline
import torch
import os

prompt = "a watercolor painting of beautiful sks woman, vibrant colors"

# Define user and model
user_id = "user_1"
model_id = "test_model"

# Define output directory
output_dir = f"./data/output/{user_id}/{model_id}/"
os.makedirs(output_dir, exist_ok=True)

# Count existing images in the output directory
existing_images = [file for file in os.listdir(output_dir) if file.endswith(('.png', '.jpg', '.jpeg'))]
image_id = len(existing_images) + 1  # Assign ID as n+1

# Load the trained model
pipeline = DiffusionPipeline.from_pretrained(
    f"./trained_model/{user_id}/{model_id}",
    torch_dtype=torch.float16,
    use_safetensors=True
).to("cuda")

# Generate image
image = pipeline(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

# Save the image with the new ID
image_path = os.path.join(output_dir, f"image_{image_id}.png")
image.save(image_path)

print(f"✅ Image saved as {image_path}")

