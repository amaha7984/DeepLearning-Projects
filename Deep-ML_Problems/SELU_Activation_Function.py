#Problem 103: Implement the SELU Activation Function
# -----------------------------------------------------
import numpy as np

def selu(x: float) -> float:
	alpha = 1.6732632423543772
	scale = 1.0507009873554804

	if x ==0:
		return 0.0

	elif x < 0:
		return alpha * scale * (np.exp(x) - 1)

	else:
		return scale * x