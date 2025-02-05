import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

@st.cache_resource
def load_clip_model():
    model_url = "https://tfhub.dev/google/clip/1"
    model = hub.load(model_url)
    return model

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return image

def classify_with_clip(image, learned_prompts):
    model = load_clip_model()
    processed_image = preprocess_image(image)
    
    # Add batch dimension
    image_input = tf.expand_dims(processed_image, 0)
    
    # Get image embeddings
    image_embeddings = model.signatures['image_embeddings'](tf.constant(image_input))['image_embeddings']
    
    # Calculate similarities
    similarities = tf.matmul(image_embeddings, learned_prompts, transpose_b=True)
    
    # Apply softmax to get probabilities
    probs = tf.nn.softmax(similarities, axis=-1)
    
    candidate_labels = ["pan", "no pan"]
    best_idx = tf.argmax(probs[0]).numpy()
    confidence = float(probs[0][best_idx])
    label = candidate_labels[best_idx]
    
    return label, confidence

@st.cache_resource
def load_learned_prompts():
    return np.load("/workspace/gourmetfoodclassifierv1.2/learned_prompts.npy")

st.title("Clasificador de Pan vs. No Pan")
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen subida", use_column_width=True)
    if st.button("Clasificar"):
        with st.spinner("Clasificando..."):
            learned_prompts = load_learned_prompts()
            label, confidence = classify_with_clip(image, learned_prompts)
        st.success(f"Predicción: {label} con una confianza del {confidence*100:.2f}%")

