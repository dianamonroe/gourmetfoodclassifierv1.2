import os
from flask import Flask, render_template, request, url_for
from pyngrok import ngrok
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np

app = Flask(__name__, template_folder='templates', static_folder='static') 
os.makedirs('static', exist_ok=True)

# Paths
learned_prompts_path = 'learned_prompts.pt'  # Adjust to the correct path on your server

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
learned_prompts = torch.load(learned_prompts_path)  
candidate_labels = ["pan", "no pan"]

# Function to classify images
def classify_with_clip(image_path):
    """Classifies an image using CLIP with learned prompts."""
    image = Image.open(image_path).convert("RGB")
    image_inputs = processor(images=image, return_tensors="pt", padding=True)
    image_features = model.get_image_features(**image_inputs)
    similarities = torch.matmul(image_features, learned_prompts.t())
    probs = torch.softmax(similarities, dim=1).cpu().numpy()[0]
    best_idx = int(np.argmax(probs))
    confidence = float(probs[round(best_idx)])
    label = candidate_labels[best_idx]
    return {"label": label, "confidence": confidence}

# Route for handling file uploads and predictions
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part" 
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file:
            # Save uploaded image
            filename = file.filename
            file_path = os.path.join('static', filename)
            file.save(file_path)
            # Classify image and get prediction
            result = classify_with_clip(file_path)
            # Render HTML with prediction and image
            return render_template('index.html', 
                                   prediction=f"I'm {result['confidence']*100:.2f}% sure this is {result['label']}",
                                   img_path=url_for('static', filename=filename))
    # Render the initial HTML template
    return render_template('index.html')  

if __name__ == '__main__':
    port = 5000 
    public_url = ngrok.connect(port).public_url
    print(f" * Running on {public_url}") 
    app.run(port=port)