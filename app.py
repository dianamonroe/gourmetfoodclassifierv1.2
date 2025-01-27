import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load model
@st.cache_resource
def load_model():
    model_path = os.path.join("models", "Yolov86thRoundbestWeights.pt")
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        logging.error(f"Model file not found: {model_path}")
        return None
    try:
        # Load YOLO model (ensures it works with CPU)
        model = YOLO(model_path)
        model.overrides["device"] = "cpu"  # Force CPU mode
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        logging.error(f"Error loading model: {e}")
        return None

# Streamlit UI
st.title("Bread Classifier App")
st.markdown("Upload an image and click 'Classify' to determine if it contains bread.")

# Upload area
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Classify button
    if st.button("Classify"):
        model = load_model()
        if model:
            try:
                # Run inference
                results = model.predict(image, imgsz=640, conf=0.25, device="cpu")

                # Parse results
                result = results[0]
                if result.probs:
                    confidence = result.probs.top1conf * 100
                    classification = result.names[result.probs.top1]

                    # Display result
                    if classification == "bread":
                        st.success(f"I'm {confidence:.2f}% sure this is bread.")
                    else:
                        st.warning(f"I'm {confidence:.2f}% sure this is not bread.")
                else:
                    st.error("Unable to classify the image.")
            except Exception as e:
                st.error(f"Error during classification: {e}")
                logging.error(f"Error during classification: {e}")
        else:
            st.error("Model failed to load. Check logs for details.")
