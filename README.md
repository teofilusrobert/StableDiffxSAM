# StableDiffxSAM
Below is a step-by-step guide to creating an AI API with Stable Diffusion using Google Colab. This will allow you to generate images from text using the Stable Diffusion model and expose the functionality via an API.

## Step 1: Set Up Google Colab Environment
Open Google Colab: Go to Google Colab.
Create a New Notebook: Click on "File" → "New Notebook".
## Step 2: Install Required Libraries
First, install the necessary Python libraries. Run the following in a Colab cell:

python
Copy code
```
!pip install diffusers transformers accelerate torch flask
```
## Step 3: Load the Stable Diffusion Model
Next, you need to load the Stable Diffusion model from Hugging Face. Use the following code:

python
Copy code
```
from diffusers import StableDiffusionPipeline
import torch

# Load the model
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
pipe = pipe.to("cuda")
```
## Step 4: Create the API Using Flask
You will create a simple Flask API to interact with the Stable Diffusion model. Add the following code to set up the API:

python
Copy code
```
from flask import Flask, request, jsonify
from PIL import Image
import base64
from io import BytesIO

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate_image():
    try:
        data = request.json
        prompt = data.get("prompt", "")
        
        # Generate the image
        image = pipe(prompt).images[0]
        
        # Convert image to base64 string
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return jsonify({"image": img_str})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```
## Step 5: Run the API in Google Colab
To run the Flask API in Google Colab, add the following code to execute the server:

python
Copy code
```
!nohup flask run --host=0.0.0.0 --port=5000 &
```
This will start the Flask server in the background.

## Step 6: Test the API
You can test the API by sending a POST request with a JSON payload containing the prompt. Here's how you can do it from within Colab:

python
Copy code
```
import requests

url = 'http://127.0.0.1:5000/generate'
data = {'prompt': 'a futuristic cityscape at sunset'}

response = requests.post(url, json=data)

# Decode the image from base64
img_data = base64.b64decode(response.json()['image'])
img = Image.open(BytesIO(img_data))
img.show()
```
## Step 7: Expose the API
Since Google Colab does not provide a public URL by default, you can use a service like ngrok to expose your API. Install ngrok:

python
Copy code
```
!pip install pyngrok
from pyngrok import ngrok
```
# Start ngrok tunnel
```
public_url = ngrok.connect(5000)
print(f"Public URL: {public_url}")
```
Now, your API is accessible via the public URL provided by ngrok. You can send requests to this URL from anywhere.

Step 8: Save and Share the Notebook
Save the Notebook: Click on "File" → "Save a copy in Drive" to save it to your Google Drive.
Share the Notebook: To share, click on "Share" in the top-right corner of the Colab notebook, set the permissions, and send the link to others.
Final Notes
Keep in mind that this is a basic setup and might require further enhancements for production use.
Stable Diffusion models are resource-intensive, so ensure your Colab environment has sufficient resources.
Now you have a functional API for generating images with Stable Diffusion, running on Google Colab!
