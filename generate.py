import base64
import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
API_URL = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-refiner-1.0"
headers = {"Authorization": f"Bearer {API_TOKEN}"}
print(headers)

def query(payload):
    with open(payload["inputs"], "rb") as f:
        img = f.read()
        payload["inputs"] = base64.b64encode(img).decode("utf-8")
    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        with open("generated_image.png", "wb") as f:
            f.write(response.content)
        print("✅ Image generated successfully!")
    else:
        print(f"❌ Error: {response.text}")

    return response.content

image_bytes = query({
    "inputs": "./pic.png",
    "parameters": {
        "prompt": "A hyper-realistic photo of a black shar pei with gold chains, color palette background, cinematic lighting, ultra-detailed, 4K",
        "negative_prompt": "blurry, distorted, deformed, low quality, extra limbs",
        "strength": 0.75,
        "guidance_scale": 7.5,
        "num_inference_steps": 50
    }
})
