'''
-*- coding: utf-8 -*-

U-Net module.

Author: Rayykia
'''

import torch
from torch import Tensor
import torch.nn as nn



import numpy as np

def ConvReLU(
        inplanes: int,
        outplanes: int,
        inplace: bool = False
) -> nn.Sequential:
    '''
    Conv3x3 + ReLU.
    Conv3x3 + ReLU.
    '''
    return nn.Sequential(
        nn.Conv2d(
            in_channels=inplanes,
            out_channels=outplanes,
            kernel_size=(3,3),
            padding=1
        ),
        nn.ReLU(inplace),
        nn.Conv2d(
            in_channels=outplanes,
            out_channels=outplanes,
            kernel_size=(3,3),
            padding=1
        ),
        nn.ReLU(inplace)
    )

class DownsampleBlock(nn.Module):
    def __init__(
            self,
            inplanes: int,
            outplanes: int,
    ) -> None:
        super(DownsampleBlock, self).__init__()
        self.conv_relu = ConvReLU(inplanes, outplanes)
        self.downsample = nn.MaxPool2d(kernel_size=(2,2))

    def forward(
            self, 
            x: Tensor, 
            downsample: bool = True
    ) -> Tensor:
        if downsample:
            x = self.downsample(x)
        x = self.conv_relu(x)
        return x
    

class UpsampleBlock(nn.Module):
    def __init__(
            self,
            planes: int
    ) -> None:
        super(UpsampleBlock, self).__init__()
        self.conv_relu = ConvReLU(planes*2, planes, inplace=True)
        self.upconv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=planes,
                out_channels=planes//2,
                kernel_size=(3,3),
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.ReLU()
        )
            
    def forward(
            self,
            x: Tensor,
    ) -> Tensor:
        x = self.conv_relu(x)
        x = self.upconv(x)
        return (x)
    

class UNetModel(nn.Module):
    def __init__(self) -> None:
        super(UNetModel, self).__init__()
        # downsample
        self.down0 = DownsampleBlock(3, 64)
        self.down1 = DownsampleBlock(64, 128)
        self.down2 = DownsampleBlock(128, 256)
        self.down3 = DownsampleBlock(256, 512)
        self.down4 = DownsampleBlock(512, 1024)

        # upsample
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=1024,
                out_channels=512,
                kernel_size=(3,3),
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.ReLU()
        )
        self.up2 = UpsampleBlock(512)
        self.up3 = UpsampleBlock(256)
        self.up4 = UpsampleBlock(128)

        self.conv_relu = ConvReLU(128, 64, inplace=True)
        self.segmentation = nn.Conv2d(
            in_channels=64,
            out_channels=2,
            kernel_size=(1,1),
        )

    def forward(
            self,
            x: Tensor
    ) -> Tensor:
        # downsample
        x0 = self.down0(x, downsample=False)            # 64*256*256
        x1 = self.down1(x0)                             # 128*128*128
        x2 = self.down2(x1)                             # 256*64*64
        x3 = self.down3(x2)                             # 512*32*32
        x4 = self.down4(x3)                             # 1024*16*16

        # upsample
        out = self.up1(x4)                              # 512*32*32
        out = self.up2(torch.cat([x3, out], dim=1))     # 1024*32*32 -> 256*64*64
        out = self.up3(torch.cat([x2, out], dim=1))     # 512*64*64 -> 128*128*128
        out = self.up4(torch.cat([x1, out], dim=1))     # 64*256*256 ->

        # segmentation
        out = self.conv_relu(torch.cat([x0, out], dim=1))
        out = self.segmentation(out)

        return out


# Print the model.
# model = UNetModel()
# print(model)