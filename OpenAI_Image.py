import openai
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

# Replace with your OpenAI API key
openai.api_key = 'YOUR_API_KEY'

def generate_image(prompt):
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="512x512"
    )
    image_url = response['data'][0]['url']

    # Fetch and display the image
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# Example prompt
prompt = "a futuristic cityscape with flying cars"
generate_image(prompt)
