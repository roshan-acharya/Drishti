import streamlit as st
from PIL import Image
import torch
from neuralnets.generator import Generator
import torchvision.transforms as transforms

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)
generator.load_state_dict(torch.load("models/generator.pth", map_location=device))
generator.eval()

# Preprocess image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(image).unsqueeze(0).to(device)

# Postprocess image
def postprocess_image(tensor):
    tensor = tensor.squeeze(0).cpu().detach()
    tensor = (tensor + 1) / 2  # Denormalize to [0,1]
    image = transforms.ToPILImage()(tensor)
    return image

# UI
st.title("Drishti - Seeing SAR Clearly")
st.write("Upload a SAR image to generate its optical image.")

uploaded_file = st.file_uploader("Choose a SAR image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Generate optical image
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        output_tensor = generator(input_tensor)
    output_image = postprocess_image(output_tensor)
    
    # Display side by side
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption='Uploaded SAR Image', width='content')
    with col2:
        st.image(output_image, caption='Generated Optical Image', width='content')
