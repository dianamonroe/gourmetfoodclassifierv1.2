import streamlit as st
from PIL import Image
import numpy as np
import onnxruntime as ort

# Set the confidence threshold for classification
confidence_level = 0.25

@st.cache_resource
def load_model():
    # Specify the path to your ONNX model
    model_path = "models/Yolov86thRoundbestWeights.onnx"
    try:
        # Create an ONNX Runtime InferenceSession
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        return session
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

def preprocess_image(image):
    # Resize the image to match the model's expected input size
    image = image.resize((640, 640))
    # Convert the image to a numpy array and normalize pixel values
    image = np.array(image, dtype=np.float32) / 255.0
    # Transpose the image to match the model's expected input shape (NCHW)
    image = np.transpose(image, (2, 0, 1))
    # Add a batch dimension
    image = np.expand_dims(image, axis=0)
    return image

def postprocess_predictions(predictions):
    try:
        # Extract the output from the ONNX model
        # The shape of the output tensor is typically [1, num_classes, grid_size, grid_size]
        output = predictions[0]
        
        # Reshape the output to [grid_size * grid_size, num_classes]
        reshaped_output = output.reshape(output.shape[1], -1).T
        
        # Get the class with the highest probability for each grid cell
        class_predictions = np.argmax(reshaped_output, axis=1)
        
        # Count the occurrences of each class
        class_counts = np.bincount(class_predictions, minlength=2)
        
        # Determine the final class based on the majority vote
        final_class = np.argmax(class_counts)
        
        # Calculate the confidence as the proportion of grid cells predicting the final class
        confidence = class_counts[final_class] / len(class_predictions)
        
        if confidence >= confidence_level:
            if final_class == 0:  # Assuming class index 0 corresponds to "bread"
                result = "Cool! I can go on analysing this image as bread above or below 90% sourdough"
            else:
                result = "Yeigs! The quality and characteristics of this image doesn't allow me to further analyse if it is bread above or below 90% sourdough"
        else:
            result = "The confidence level is too low for me to further analyse this image"

        return result
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
            # Preprocess the image
            input_image = preprocess_image(image)
            
            # Run inference
            input_name = model.get_inputs()[0].name
            output_name = model.get_outputs()[0].name
            predictions = model.run([output_name], {input_name: input_image})
            
            # Postprocess the predictions
            result = postprocess_predictions(predictions)
            st.success(result)
        else:
            st.error("Failed to load model.")

