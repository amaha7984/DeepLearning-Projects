#-----------Single Neuron Operation with three inputs-----------#
# Each input has a weight, so total three weights
# each neuron only has only one bias
"""
inputs = [1, 2, 3]
weights = [0.2, 0.8, -0.5]
bias = 2

# f(x) = w^T * x + b
outputs = []
for i in range(len(inputs)):
    outputs.append(inputs[i] * weights[i])
x = 0
for j in range(len(outputs)):
    x += outputs[j]
    j += 1
print(x+bias)


#using zip in for loop

outputs = 0
for i, j in zip(inputs, weights):
    outputs += i * j

final_output = outputs + bias
print(final_output)


################################################
#----------Three neurons and four weights----------------#

inputs = [1, 2, 3, 2.5]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]
layer_outputs = [] 
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0 #output of current neuron

    for weight, input in zip(neuron_weights, inputs):
        neuron_output += weight * input
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)
print(layer_outputs)


################################################
#----------Three neurons and four weights with numpy dot product----------------#

import numpy as np

inputs = [1, 2, 3, 2.5] #shape is (4,)

weights = [[0.2, 0.8, -0.5, 1.0],  #shape is (3, 4)
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

print(np.dot(weights, inputs) + biases)

################################################
#----------Batches of Samples----------------#

import numpy as np

inputs = [[1, 2, 3, 2.5],
          [2, 5, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]] #shape is (3, 4)

weights = [[0.2, 0.8, -0.5, 1.0],  #shape is (3, 4)
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

output = np.dot(inputs, np.array(weights).T) + biases
print(output)

"""
################################################
#----------Layers and Objects with Hidden Layer Activation Function----------------#

import numpy as np 
import nnfs #pip install nnfs
from nnfs.datasets import spiral_data 

nnfs.init()

np.random.seed(0)

X, y = spiral_data(100, 3)  

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        self.output =  (np.dot(inputs, self.weights)) + self.biases

class Relu_Activation:
    def forward(self, numbers):
        self.output = np.maximum(0, numbers)

layer1 = Layer_Dense(2, 5)

relu1 = Relu_Activation()

layer1.forward(X)
# print(layer1.output)
relu1.forward(layer1.output)
print(relu1.output)

