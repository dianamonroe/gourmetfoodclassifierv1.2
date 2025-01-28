import streamlit as st
from PIL import Image
import numpy as np
import onnxruntime as ort

# Set the confidence threshold for classification
confidence_level = 0.10

@st.cache_resource
def load_model():
    model_path = "models/Yolov86thRoundbestWeights.onnx"
    try:
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        return session
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

def preprocess_image(image):
    image = image.resize((640, 640))
    image = np.array(image, dtype=np.float32) / 255.0
    image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, axis=0)
    return image

def postprocess_predictions(predictions):
    try:
        # YOLOv8 ONNX model typically outputs a tensor of shape [1, 84, 8400]
        # where 84 = 4 (bbox) + 1 (confidence) + 79 (class probabilities)
        output = predictions[0]
        
        # Reshape the output to [8400, 84]
        output = output.squeeze().T
        
        # Extract confidence scores and class probabilities
        confidence = output[:, 4]
        class_probs = output[:, 5:]
        
        # Get the class with highest probability for each detection
        class_ids = np.argmax(class_probs, axis=1)
        
        # Filter detections based on confidence threshold
        mask = confidence > confidence_level
        filtered_confidence = confidence[mask]
        filtered_class_ids = class_ids[mask]
        
        if len(filtered_class_ids) > 0:
            # Count occurrences of each class
            class_counts = np.bincount(filtered_class_ids)
            final_class = np.argmax(class_counts)
            confidence = np.mean(filtered_confidence)
            
            if final_class == 0:  # Assuming class index 0 corresponds to "bread"
                result = f"Cool! I can go on analysing this image as bread above or below 90% sourdough (Confidence: {confidence:.2f})"
            else:
                result = f"Yeigs! The quality and characteristics of this image doesn't allow me to further analyse if it is bread above or below 90% sourdough (Confidence: {confidence:.2f})"
        else:
            result = "The confidence level of my prediction is too low for me to further analyse this image"


        return result
    except Exception as e:
        return f"Error during classification: {str(e)}"

# Streamlit app
st.title("Bread Classifier App")
st.markdown("Upload an image and click 'Classify' to determine if it can be classified as gourmet bread.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Classify"):
        model = load_model()
        if model:
            input_image = preprocess_image(image)
            
            # Get input and output names
            input_name = model.get_inputs()[0].name
            output_name = model.get_outputs()[0].name
            
            # Run inference
            predictions = model.run([output_name], {input_name: input_image})
            
            result = postprocess_predictions(predictions)
            st.success(result)
        else:
            st.error("Failed to load model.")

