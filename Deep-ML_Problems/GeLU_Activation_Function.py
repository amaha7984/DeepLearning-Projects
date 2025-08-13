# Problem: 147 GeLU Activation Function
#-------------------------------------

#Numpy Solution
import numpy as np

def GeLU(x: np.ndarray) -> np.ndarray:
	# Your code here
    scores = 0.5 * x * ( 1.0 + np.tanh(np.sqrt(2.0/np.pi) * (x + 0.044715 * (x**3))))


	return np.round(scores, 4)