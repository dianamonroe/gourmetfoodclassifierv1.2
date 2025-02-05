import os
from flask import Flask, render_template, request, jsonify
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

app = Flask(__name__)

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load learned prompts
learned_prompts_path = "/workspace/gourmetfoodclassifierv1.2/learned_prompts.pt"
learned_prompts = torch.load(learned_prompts_path, map_location=torch.device('cpu'))

def classify_with_clip(image):
    inputs = processor(images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    similarities = torch.matmul(image_features, learned_prompts.t())
    probs = torch.softmax(similarities, dim=1).cpu().numpy()[0]
    candidate_labels = ["pan", "no pan"]
    best_idx = int(torch.argmax(probs))
    confidence = float(probs[best_idx])
    label = candidate_labels[best_idx]
    return label, confidence

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        if file:
            image = Image.open(file.stream).convert("RGB")
            label, confidence = classify_with_clip(image)
            return jsonify({'label': label, 'confidence': confidence})
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))

