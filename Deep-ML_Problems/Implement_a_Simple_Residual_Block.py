# Problem 113: Implement a Simple Residual Block with Shortcut Connection

import numpy as np

def relu(x):
	if x <= 0:
		return 0
	else:
		return x 

def residual_block(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    first_array = []
	second_array = []
	first = np.matmul(w1, x)
	for i in first:
		first_array.append(relu(i))

	second = np.add(np.matmul(w2, first), x)
	for j in second:
		second_array.append(relu(j))
	return second_array