import streamlit as st
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import pickle

@st.cache_resource
def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    # Load learned_prompts
    learned_prompts_path = "/workspace/gourmetfoodclassifierv1.2/learned_prompts.npy"
    learned_prompts = np.load(learned_prompts_path)
    return model, processor, learned_prompts

def classify_with_clip(image):
    model, processor, learned_prompts = load_clip_model()
    inputs = processor(images=image, return_tensors="np", padding=True)
    image_features = model.get_image_features(**inputs)
    image_features = image_features.detach().numpy()
    similarities = np.matmul(image_features, learned_prompts.T)
    probs = np.exp(similarities) / np.sum(np.exp(similarities), axis=1, keepdims=True)
    probs = probs[0]
    candidate_labels = ["pan", "no pan"]
    best_idx = int(np.argmax(probs))
    confidence = float(probs[best_idx])
    label = candidate_labels[best_idx]
    return label, confidence

st.title("Clasificador de Pan vs. No Pan")
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen subida", use_column_width=True)
    if st.button("Clasificar"):
        with st.spinner("Clasificando..."):
            label, confidence = classify_with_clip(image)
        st.success(f"Predicción: {label} con una confianza del {confidence*100:.2f}%")

