import streamlit as st
from PIL import Image
import numpy as np
import onnxruntime as ort
import io

# Set the confidence threshold for classification
confidence_level = 0.10

@st.cache_resource
def load_model():
    model_path = "models/best.onnx"
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
        output = predictions[0]
        output = output.squeeze().T
        confidence = output[:, 4]
        class_probs = output[:, 5:]
        class_ids = np.argmax(class_probs, axis=1)
        mask = confidence > confidence_level
        filtered_confidence = confidence[mask]
        filtered_class_ids = class_ids[mask]
        
        if len(filtered_class_ids) > 0:
            class_counts = np.bincount(filtered_class_ids)
            final_class = np.argmax(class_counts)
            confidence = np.mean(filtered_confidence)
            
            if final_class == 0:
                result = f"Cool! I can analyze this image as bread. Confidence: {confidence * 100:.2f}"
            else:
                result = f"This image doesn't appear to be bread. Confidence: {confidence * 100:.2f}"
        else:
            result = "The confidence level is too low to analyze this image"

        return result
    except Exception as e:
        return f"Error during classification: {str(e)}"

# Streamlit app
st.title("Bread Classifier App")
st.markdown("Upload an image and click 'Classify' to determine if it can be classified as gourmet bread.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Read the file into bytes
        image_bytes = uploaded_file.read()
        
        # Open the image using PIL
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Display the image
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Classify button
        if st.button("Classify"):
            model = load_model()
            if model:
                input_image = preprocess_image(image)
                
                input_name = model.get_inputs()[0].name
                output_name = model.get_outputs()[0].name
                predictions = model.run([output_name], {input_name: input_image})
                
                result = postprocess_predictions(predictions)
                st.success(result)
            else:
                st.error("Failed to load model.")
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
else:
    st.info("Please upload an image to classify.")

