import streamlit as st
from PIL import Image
import numpy as np
import onnxruntime as ort

# Load ONNX model
@st.cache_resource
def load_model():
    model_path = "models/Yolov86thRoundbestWeights.onnx"
    try:
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        return session
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# Preprocess image
def preprocess_image(image):
    image = image.resize((640, 640))  # Resize to Yolov8 input size
    image = np.array(image, dtype=np.float32)
    image = np.transpose(image, (2, 0, 1))  # HWC to CHW
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image /= 255.0  # Normalize to [0, 1]
    return image

# Postprocess predictions
def postprocess_predictions(predictions):
    """
    Process YOLO predictions to return a simple classification result.
    """
    try:
        # Extract the first output tensor (class probabilities)
        probs = predictions[0][0]  # Assuming a batch of size 1

        # Get the top predicted class and its confidence
        class_idx = probs.argmax()  # Index of top class
        confidence = probs[class_idx] * 100
        label = "bread" if class_idx == 1 else "not bread"

        return f"I'm {confidence:.2f}% sure this is {label}."
    except Exception as e:
        return f"Error during classification: {str(e)}"

# Streamlit app
st.title("Bread Classifier App")
st.markdown("Upload an image and click 'Classify' to determine if it can be classified as gourmet bread.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    # Display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Classify button
    if st.button("Classify"):
        model = load_model()
        if model:
            input_image = preprocess_image(image)
            predictions = model.run(None, {"images": input_image})
            result = postprocess_predictions(predictions)
            st.success(result)
        else:
            st.error("Failed to load model.")
