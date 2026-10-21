import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pytorch_lightning as pl
from torch.nn import CrossEntropyLoss
import sys

# Configure page metadata and theme
st.set_page_config(page_title="Fashion Classifier App", page_icon="👗", layout="centered")

# Standard Fashion-MNIST Class Categories
FASHION_CLASSES = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boots"
}

# 1. Custom Core Sequential Model (MyMNISTModel)
class MyMNISTModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 8),  # Matching your 8 hidden layer nodes configuration
            nn.ReLU(),
            nn.Linear(8, 10)
        )
    def forward(self, x):
        return self.model(x)

# Bind the class object reference to the primary main namespace
sys.modules['__main__'].MyMNISTModel = MyMNISTModel

# 2. Main PyTorch Lightning Module
class DigitClassifier(pl.LightningModule):
    def __init__(self, architecture=None, lr=1e-3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(ignore=['architecture'])
        self.model = architecture if architecture is not None else MyMNISTModel()
        self.lr = lr
        self.loss_fn = CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

@st.cache_resource
def load_trained_model_final():
    # Hardcoded to your actual Google Colab path
    checkpoint_path = "epoch=175-step=32912.ckpt"
    try:
        model = DigitClassifier.load_from_checkpoint(
            checkpoint_path,
            architecture=MyMNISTModel(),
            lr=1e-3,
            weights_only=False
        )
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image):
    # Synced 1:1 with your high-precision non-web pipeline settings
    preprocess = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])
    return preprocess(image).unsqueeze(0)

# --- UI Front-end Layout Setup ---
st.title("👗 Fashion Classifier Web App")
st.write("Upload an image of clothing, apparel, or footwear to let the AI classify the product category automatically.")

model = load_trained_model_final()
if model is not None:
    st.success("🎉 Loaded model complete (Synced with high-precision pipeline)")
else:
    st.warning("⚠️ Warning: The checkpoint is None. Please contact the developer.")

# User Guidelines Accordion
with st.expander("💡 Tips for the Best Prediction Accuracy"):
    st.write("- Use clear images with **only one product** centered in the frame.")
    st.write("- Ensure the item has a visible contrast against its background.")
    st.write("- Supported file formats: **.png, .jpg, .jpeg**")

st.write("---")

uploaded_file = st.file_uploader("Choose image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # Split UI view into 2 clean columns (Left: Uploaded Image | Right: Dynamic Prediction Stats)
    col1, col2 = st.columns(2)

    # Enforce standard RGB mode reading to mirror PILImage.create behavior
    image = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.image(image, caption="📷 Uploaded Image", use_container_width=True)

    img_tensor = preprocess_image(image)

    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1).squeeze(0)

        pred_idx = output.argmax(dim=1).item()
        confidence = probabilities[pred_idx].item() * 100

    class_name = FASHION_CLASSES.get(pred_idx, "Unknown Category")

    with col2:
        st.subheader("📊 Classification Results:")
        st.metric(label="🎯 Result (Prediction)", value=class_name)

        # Real-time Confidence Meter Progress Bar
        st.write(f"📈 **Confidence:** {confidence:.2f}%")
        st.progress(int(confidence))

    st.write("---")

    # Clean probability metric graph view with no tables attached
    st.write("🔍 **The Graph of Probabilities:**")
    chart_data = {FASHION_CLASSES[i]: (float(probabilities[i]) * 100) for i in range(10)}
    st.bar_chart(chart_data)

# Footer Area Credits
st.write("---")
st.caption("🤖 Powered by PyTorch Lightning & Streamlit | Developed by AI Collaborator")
