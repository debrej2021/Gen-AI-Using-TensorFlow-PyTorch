from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import matplotlib.pyplot as plt

# Load the pipeline
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to(device)

def generate_image(prompt):
    with torch.autocast(device):
        image = pipe(prompt)["sample"][0]
    # Display the image
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# Example prompt
prompt = "a futuristic cityscape with flying cars"
generate_image(prompt)
