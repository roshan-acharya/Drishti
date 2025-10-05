#Dataset and Dataloader
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import os
import matplotlib.pyplot as plt

class SARDataset(Dataset):
    def __init__(self, sar_dir, opt_dir, transform=None):
        self.sar_dir=sar_dir
        self.opt_dir=opt_dir
        self.transform=transform

        self.sar_images=sorted(os.listdir(sar_dir))
        self.opt_images=sorted(os.listdir(opt_dir))

        if len(self.sar_images) != len(self.opt_images):
            raise ValueError("The number of SAR and optical images must be the same.")

    def __len__(self):
        return len(self.sar_images)
    
    def __getitem__(self, idx):
        sar_image_path=os.path.join(self.sar_dir,self.sar_images[idx])
        opt_image_path=os.path.join(self.opt_dir,self.opt_images[idx])

        sar_image=Image.open(sar_image_path).convert("RGB")
        opt_image=Image.open(opt_image_path).convert("RGB")

        if self.transform:
            sar_image=self.transform(sar_image)
            opt_image=self.transform(opt_image)

        return sar_image,opt_image
    
#transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    #No need to calculate mean of image as it is done to match tanh output in GAN
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def get_dataloaders(sar_dir, opt_dir, sar_val_dir, opt_val_dir, batch_size=16):
    dataset = SARDataset(sar_dir, opt_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    val_dataset = SARDataset(sar_val_dir, opt_val_dir, transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    return dataloader, val_dataloader