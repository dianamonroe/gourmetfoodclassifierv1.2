import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Load YOLO model
model = YOLO('models/Yolov86thRoundbestWeights.pt')

# App Title
st.title("Food Image Classifier")

# File Upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display uploaded image
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Make prediction
    results = model(img)
    st.write("Predictions:")
    for result in results:
        st.write(f"Class: {result.name}, Confidence: {result.confidence:.2f}")
