import streamlit as st
from PIL import Image
import torch
import torch.nn as nn



import torchvision.transforms as transforms
from huggingface_hub import hf_hub_download

#Generator
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super().__init__()

        #Downsampling code

        if down:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2) if act=="leaky" else nn.ReLU()
            )

        #Upsampling code

        else: 
            self.block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(0.5) if use_dropout else nn.Identity(),
                nn.ReLU()
            )
    def forward(self, x):
        return self.block(x)
    
class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1, padding_mode="reflect"), # 128x128
            nn.LeakyReLU(0.2)
        )

        #Encoder
        self.down1 = Block(features, features*2, down=True, act="leaky", use_dropout=False) # 64*64
        self.down2 = Block(features*2, features*4, down=True, act="leaky", use_dropout=False) # 32*32
        self.down3 = Block(features*4, features*8, down=True, act="leaky", use_dropout=False) # 16*16
        self.down4 = Block(features*8, features*8, down=True, act="leaky", use_dropout=False) # 8*8
        self.down5 = Block(features*8, features*8, down=True, act="leaky", use_dropout=False) # 4*4
        self.down6 = Block(features*8, features*8, down=True, act="leaky", use_dropout=False) # 2*2

        #bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8, features*8, kernel_size=4, stride=2, padding=1, padding_mode="reflect"), #1x1
            nn.ReLU()
        )

        #decoder
        self.up1 = Block(features*8, features*8, down=False, act="relu", use_dropout=True) #2x2
        self.up2 = Block(features*8*2, features*8, down=False, act="relu", use_dropout=True) #4x4
        self.up3 = Block(features*8*2, features*8, down=False, act="relu", use_dropout=True) #8x8
        self.up4 = Block(features*8*2, features*8, down=False, act="relu", use_dropout=False) #16x16
        self.up5 = Block(features*8*2, features*4, down=False, act="relu", use_dropout=False) #32x32
        self.up6 = Block(features*4*2, features*2, down=False, act="relu", use_dropout=False) #64x64
        self.up7 = Block(features*2*2, features, down=False, act="relu", use_dropout=False) #128x128

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features*2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self,x):
            d1 = self.initial_down(x)
            d2 = self.down1(d1)
            d3 = self.down2(d2)
            d4 = self.down3(d3)
            d5 = self.down4(d4)
            d6 = self.down5(d5)
            d7 = self.down6(d6)
            
            bottleneck = self.bottleneck(d7)
            
            up1 = self.up1(bottleneck)
            up2 = self.up2(torch.cat([up1, d7], 1))
            up3 = self.up3(torch.cat([up2, d6], 1))
            up4 = self.up4(torch.cat([up3, d5], 1))
            up5 = self.up5(torch.cat([up4, d4], 1))
            up6 = self.up6(torch.cat([up5, d3], 1))
            up7 = self.up7(torch.cat([up6, d2], 1))
            
            return self.final_up(torch.cat([up7, d1],1))




# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)
model_path = hf_hub_download(repo_id="roshan-acharya/dristhi-generator", filename="generator.pth")
generator.load_state_dict(torch.load(model_path, map_location=device))

generator.eval()

# Preprocess & postprocess
# Preprocess & postprocess
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(image).unsqueeze(0).to(device)

def postprocess_image(tensor):
    tensor = tensor.squeeze(0).cpu().detach()
    tensor = (tensor + 1) / 2
    return transforms.ToPILImage()(tensor)
    tensor = (tensor + 1) / 2
    return transforms.ToPILImage()(tensor)

# Streamlit UI
# Streamlit UI
st.title("Drishti - Seeing SAR Clearly")
st.write("Upload a SAR image to generate its optical image.")

uploaded_file = st.file_uploader("Choose a SAR image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.write("Generating optical image...")


    input_tensor = preprocess_image(image)
    with torch.no_grad():
        output_tensor = generator(input_tensor)
    output_image = postprocess_image(output_tensor)


    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption='Uploaded SAR Image', width='stretch')
    with col2:
        st.image(output_image, caption='Generated Optical Image', width='stretch')
