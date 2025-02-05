import streamlit as st
from PIL import Image
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel

@st.cache_resource
def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    # Cargar learned_prompts
    learned_prompts_path = "/content/drive/MyDrive/4Geek Final Project Pan Masa Madre 100%/Eat v1.1 -bread:not bread- Blindly Network transfer model/Open AI CLIP model bread-not bread/learned_prompts.pt"
    learned_prompts = torch.load(learned_prompts_path)  # Asumir tensor [num_prompts, D]
    return model, processor, learned_prompts

def classify_with_clip(image):
    model, processor, learned_prompts = load_clip_model()
    inputs = processor(images=image, return_tensors="pt", padding=True)
    image_features = model.get_image_features(**inputs)
    similarities = torch.matmul(image_features, learned_prompts.t())
    probs = torch.softmax(similarities, dim=1).cpu().numpy()[0]
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
        label, confidence = classify_with_clip(image)
        st.success(f"Predicción: {label} con una confianza del {confidence*100:.2f}%")
