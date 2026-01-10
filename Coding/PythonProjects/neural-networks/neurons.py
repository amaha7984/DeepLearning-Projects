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
"""

################################################
#----------Three neurons and four weights with numpy dot product----------------#

import numpy as np

inputs = [1, 2, 3, 2.5] #shape is (4,)

weights = [[0.2, 0.8, -0.5, 1.0],  #shape is (3, 4)
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

print(np.dot(weights, inputs) + biases)






    
