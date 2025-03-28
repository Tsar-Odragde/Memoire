from PIL import Image
import os

def preprocess_images(input_dir, output_dir, image_size=(512, 512)):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('png', 'jpg', 'jpeg', 'webp')):
            img_path = os.path.join(input_dir, filename)
            with Image.open(img_path) as img:
                img = img.resize(image_size)
                img.save(os.path.join(output_dir, filename))

# Example usage:
user_id = 'user_1'
model_id = 'test_model'
input_directory = f'data/input/{user_id}/{model_id}/raw_images'
output_directory = f'data/input/{user_id}/{model_id}/processed_images'
preprocess_images(input_directory, output_directory)
