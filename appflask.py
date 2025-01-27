import os
from flask import Flask, render_template, request, jsonify
from PIL import Image
import io
import torch

# Force CPU device
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.set_num_threads(1)

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model only when needed
model = None

def load_model():
    global model
    if model is None:
        try:
            from ultralytics import YOLO
            model_path = os.path.join('models', 'Yolov86thRoundbestWeights.pt')
            model = YOLO(model_path)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            model = None
    return model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Read and process the image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        
        # Save the file (optional, for debugging)
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        img.save(img_path)
        
        # Load model and make prediction
        model = load_model()
        if model is None:
            return jsonify({'error': 'Model failed to load'}), 500

        # Make prediction
        results = model(img)
        
        # Process results
        if len(results) > 0 and hasattr(results[0], 'names') and hasattr(results[0], 'probs'):
            prediction = {
                'class': results[0].names[results[0].probs.top1],
                'confidence': float(results[0].probs.top1conf)
            }
        else:
            prediction = {'error': 'Unable to process image'}
        
        return jsonify(prediction)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)