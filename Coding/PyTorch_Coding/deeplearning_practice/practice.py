"""
1. Write a Python function that computes the output of the sigmoid activation function given an input value z.
The function should return the output rounded to four decimal places.

import torch


def sigmoid(z):
    z = torch.tensor(z)
    z = 1/(1 + torch.exp(-z))
    z = z.item()
    return round(z, 4)
    
print(sigmoid(0))

######################################################################################################
2. Write a Python function that computes the softmax activation for a given list of scores.
The function should return the softmax values as a list, each rounded to four decimal places.
"""
import torch

def softmax(x):
    x = torch.tensor(x, dtype=torch.float)
    y = []
    for i in range(len(x)):
        y.append(torch.exp(x[i])/torch.sum(torch.exp(x)))
    return y


def softmax_function(x):
    x = torch.tensor(x, dtype=torch.float)
    x = softmax(x)
    y = []
    for i in range(len(x)):
        y.append(round(x[i].item(), 4))
    return y
print(softmax_function([1,2,3]))

# softmax [x[0]] = torch.exp(x[0]) / torch.sum(torch.exp(x))