import torch
import torch.nn as nn


#Generator

#Block
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
    
#UNet Model for Generator
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


