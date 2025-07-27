#Sigmoid Activation Function Understanding. Problem 22
#Write a Python function that computes the output of the sigmoid activation function given an input value z. 
#The function should return the output rounded to four decimal places

#Solution with torch
import torch

def sigmoid(z: float) -> float:
    z = torch.tensor(z, dtype=torch.float32)
	result = 1 / (1 + torch.exp(-z))
	result = round(result.item(), 4)
    return result

###################################################
#Below is solution with numpy
"""
import numpy as np

def sigmoid(z: float) -> float:
	result = round(1 / (1+np.exp(-z)), 4)
	return result
"""