# Problem 59: Implement Long Short-Term Memory (LSTM) Network
# Task: Implement Long Short-Term Memory (LSTM) Network
# The LSTM should compute the forget gate, input gate, candidate cell state, and 
# output gate at each time step to update the hidden state and cell state.

# -----------------------------------------------------------------------------------------
# Numpy Solution

import numpy as np

class LSTM:
	def __init__(self, input_size, hidden_size):
		self.input_size = input_size
		self.hidden_size = hidden_size

		# Initialize weights and biases
		self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wo = np.random.randn(hidden_size, input_size + hidden_size)

		self.bf = np.zeros((hidden_size, 1))
		self.bi = np.zeros((hidden_size, 1))
		self.bc = np.zeros((hidden_size, 1))
		self.bo = np.zeros((hidden_size, 1))

    def sigmoid(self, x):
		return 1 / (1+ np.exp(-x))

	def forget_gate(self, z):
		return self.sigmoid(np.dot(self.Wf, z) + self.bf)
	
	def input_gate(self, z):
		return self.sigmoid(np.dot(self.Wi, z) + self.bi)
	
	#Candidate cell state (also called cell input activation or new memory content)
	def candidate_state(self, z):
		return np.tanh(np.dot(self.Wc, z) + self.bc)
	
	def output_gate(self, z):
		return self.sigmoid(np.dot(self.Wo, z) + self.bo)

	def update_cellstate(self, z, initial_cell_state):
		return np.multiply(self.forget_gate(z), initial_cell_state) + np.multiply(self.input_gate(z), self.candidate_state(z))
	
	def update_hiddenstate(self,z, cell_state):
		return np.multiply(self.output_gate(z), np.tanh(cell_state))


	def forward(self, x, initial_hidden_state, initial_cell_state):
		hidden_states = []
		for i in x:
			i = i.reshape(-1, 1) 
			z = np.concatenate((initial_hidden_state, i), axis=0)
			c_res = self.update_cellstate(z, initial_cell_state)
			h_res = self.update_hiddenstate(z, c_res)
			hidden_states.append(h_res)
			initial_hidden_state = h_res
			initial_cell_state = c_res

		return hidden_states, h_res, c_res 






