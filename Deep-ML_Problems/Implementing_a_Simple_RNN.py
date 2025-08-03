# Problem 54: Implementing a Simple RNN
"""
Write a Python function that implements a simple Recurrent Neural Network (RNN) cell. The function should process a sequence of input vectors and produce the final hidden state. Use the tanh activation function for the hidden state updates. The function should take as inputs the sequence of input vectors, the initial hidden state, the weight matrices for input-to-hidden and hidden-to-hidden connections, and the bias vector. The function should return the final hidden state after processing the entire sequence, rounded to four decimal places.
"""
#---------------------------------------------------------------------------------------------------------------

#Numpy Solution
import numpy as np

def tanh(x):
	return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

def rnn_forward(input_sequence: list[list[float]], initial_hidden_state: list[float], Wx: list[list[float]], Wh: list[list[float]], b: list[float]) -> list[float]:
	for x in input_sequence:
		initial_hidden_state = tanh((np.dot(Wx, x)) + (np.dot(Wh, initial_hidden_state)) + b)


	return np.round(initial_hidden_state, 4)