import streamlit as st
from PIL import Image
import torch
from neuralnets.generator import Generator
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)
generator.load_state_dict(torch.load("models/generator.pth", map_location=device))
generator.eval()

#Preprocess image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(image).unsqueeze(0).to(device)

def postprocess_image(tensor):
    tensor = tensor.squeeze(0).cpu().detach()
    tensor = (tensor + 1) / 2  # Denormalize to [0, 1]
    image = transforms.ToPILImage()(tensor)
    return image

#UI
st.title("Image-to-Image Translation with GANs")
st.write("Upload a SAR image to generate its optical Image.")
uploaded_file = st.file_uploader("Choose a SAR image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded SAR Image')
    st.write("")
    st.write("Generating optical Image...")

    input_tensor = preprocess_image(image)
    with torch.no_grad():
        output_tensor = generator(input_tensor)
    
    output_image = postprocess_image(output_tensor)
    st.image(output_image, caption='Generated Optical Image')
