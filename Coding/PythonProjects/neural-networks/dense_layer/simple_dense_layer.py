#-------------------numpy implementation----------------------------#
"""
import numpy as np

np.random.seed(0)

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

class Dense_Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


layer_1 = Dense_Layer(4, 3)
layer_2 = Dense_Layer(3, 2)

layer_1.forward(X)
layer_2.forward(layer_1.output)
print(layer_2.output)
"""

#------------------PyTorch Implementation-------------------------------#
import torch
import torch.nn as nn


X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

class DenseLayer(nn.Module):
    def __init__(self, n_inputs, n_neurons):
        super().__init__()
        self.linear = nn.Linear(n_inputs, n_neurons)

    def forward(self, x):
        return self.linear(x)


