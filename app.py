import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from huggingface_hub import hf_hub_download
from model_arch import DeepfakeDetectorCNN 
import os
from dotenv import load_dotenv

load_dotenv()

FILENAME = os.getenv("FILENAME")
REPO_ID = os.getenv("REPO_ID")


# --- IMAGE PREPROCESSING ---
# Must match the normalization and size used during training
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

@st.cache_resource
def load_model():
    # Download weights from Hugging Face Hub
    path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    model = DeepfakeDetectorCNN()
    
    
    checkpoint = torch.load(path, map_location="cpu")
    
    # Check if checkpoint is a state_dict or a full dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    return model

# --- STREAMLIT UI ---
st.set_page_config(page_title="AI/Deepfake Detector", page_icon="🛡️")
st.title("🛡️ AI/Deepfake Detector")
st.markdown("""
    This application uses a custom Convolutional Neural Network (CNN) to detect whether a face image is **Real** or a **Deepfake/AI generated**.
""")


try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model from Hugging Face: {e}")
    st.stop()

uploaded_file = st.file_uploader("Upload a face image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Image à analyser", use_container_width=True)
    
   
    img_t = transform(image).unsqueeze(0) 
    
    with st.spinner("Analyse en cours..."):
        with torch.no_grad():
            output = model(img_t)
            # Puisque ton modèle finit par une Sigmoïde, output est déjà une probabilité (0 à 1)
            # Dans ton train : 0 = Fake, 1 = Real
            prob_real = output.item()
            prob_fake = 1 - prob_real

    
    st.divider()
    
    
    if prob_real > 0.5:
        st.success(f"✅ Prediction: **REAL**")
        st.metric("Confidence", f"{prob_real:.2%}")
        st.progress(prob_real)
    else:
        st.error(f"🚨 Prediction: **FAKE**")
        st.metric("Confidence", f"{prob_fake:.2%}")
        st.progress(prob_fake)

    
    st.info("💡 The model analyzes texture artifacts and biological inconsistencies invisible to the naked eye")