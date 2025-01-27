import streamlit as st
from PIL import Image
import os
import logging
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load model
@st.cache_resource
def load_model():
    model_path = os.path.join('models', 'Yolov86thRoundbestWeights.pt')
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        logging.error(f"Model file not found: {model_path}")
        return None
    try:
        model = YOLO(model_path)
        logging.info("Model loaded successfully")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        logging.error(f"Error loading model: {e}")
        return None

# Title and description
st.title("Food Image Classifier")
st.write("Upload an image to classify whether it's bread or not!")

# Upload area
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")

    # Load model and classify
    model = load_model()
    if model:
        try:
            results = model(image)  # Inference
            if len(results) > 0:
                st.write("### Classification Result:")
                for result in results:
                    st.write(f"- **Class**: {result.names[result.probs.top1]}")
                    st.write(f"- **Confidence**: {(result.probs.top1conf * 100):.2f}%")
            else:
                st.error("Unable to process the image.")
        except Exception as e:
            st.error(f"Error during classification: {str(e)}")
            logging.error(f"Error during classification: {str(e)}")
    else:
        st.error("Model failed to load. Check logs for details.")

def load_css(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("static/styles.css")
