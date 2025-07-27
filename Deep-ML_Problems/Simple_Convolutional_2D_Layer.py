#Problem 41 *Simple Convolutional 2D Layer*: In this problem, you need to implement a 2D convolutional layer in Python.
#This function will process an input matrix using a specified convolutional kernel, padding, and stride.

#Line 7 to 27 provides PyTorch Solution
#Line 33 to 55 provies numpy solution

import torch 
import torch.nn.functional as F

def simple_conv2d(input_matrix: torch.Tensor, kernel: torch.Tensor, padding: int, stride: int):
    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape
    
    input_matrix = F.pad(input_matrix, (padding, padding, padding, padding), mode='constant', value=0)

    output_matrix_height = ((input_matrix.shape[0] - kernel_height) // stride) + 1
    output_matrix_width = ((input_matrix.shape[1] - kernel_width) // stride) + 1

    output_matrix = torch.zeros((output_matrix_height, output_matrix_width),  dtype=input_matrix.dtype)

    for i in range(output_matrix_height):
        for j in range(output_matrix_width):
            row = i * stride
            col = j * stride
            patch = input_matrix[row:row+kernel_height, col:col+kernel_width]
            output_matrix[i, j] = torch.sum(torch.mul(patch, kernel)) 
    return output_matrix
    
###########################################################################################################
#Below is numpy solution. Uncomment to use it

"""
import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
	input_height, input_width = input_matrix.shape
	kernel_height, kernel_width = kernel.shape
    
    if padding > 0:
        input_matrix = np.pad(input_matrix, ((padding, padding), (padding, padding)), mode='constant')
    #First - (padding, padding) adds the rows on top and bottom and then, next
	#(padding, padding) adds columns on the left and right
   
    output_height = ((input_matrix.shape[0] - kernel_height) // stride) + 1
    output_width = ((input_matrix.shape[1] - kernel_width) // stride) + 1

	output_matrix = np.zeros((output_height, output_width))
    
	for i in range(0, output_height):
        for j in range(0, output_width):
            row = i * stride
            col = j * stride
            patch = input_matrix[row:row + kernel_height, col:col + kernel_width]
            output_matrix[i, j] = np.sum(np.multiply(patch, kernel))
	return output_matrix
"""