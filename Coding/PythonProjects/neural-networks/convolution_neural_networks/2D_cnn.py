import torch
import torch.nn as nn

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.conv= nn.Conv2d(in_channels, out_channels, kernel_size, padding)
    def forward(self, x):
        return self.conv(x)
   

#---------------------numpy implementation-------------------------------------#

# This is simple 2D convolution layer without channels and batch
import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):

    # O = (W - F + 2P)/S + 1

    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape

    if padding > 0:
        input_matrix = np.pad(input_matrix, ((padding, padding), (padding, padding)), mode='constant')
    
    output_height = ((input_height - kernel_height) // stride) + 1
    output_width = ((input_width - kernel_width) // stride) + 1

    output_matrix = np.zeros((output_height, output_width))

    for i in range(0, output_height):
        for j in range(0, output_width):
            row = i * stride
            col = j * stride
            patch = input_matrix[row:row+kernel_height, col:col+kernel_width]
            output_matrix[i, j]= np.sum(patch * kernel)
    
    return output_matrix


