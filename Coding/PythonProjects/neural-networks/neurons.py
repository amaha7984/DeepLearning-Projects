#-----------Single Neuron Operation with three inputs-----------#
# Each input has a weight, so total three weights
# each neuron only has only one bias
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

    
