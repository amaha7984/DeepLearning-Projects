import torch
import torch.nn as nn

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.conv= nn.Conv2d(in_channels, out_channels, kernel_size, padding)
    def forward(self, x):
        return self.conv(x)
   

