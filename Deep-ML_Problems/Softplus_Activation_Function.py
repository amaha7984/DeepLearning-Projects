# Problem 99: Implement the Softplus Activation Function
# -------------------------------------------------------

import numpy as np

def softplus(x: float) -> float:
	if x < 0:
		val = - 1 * x
		val = np.log(1 + np.exp(x))
		return round(val,4)
	else:
		val = np.log(1 + np.exp(x))
		return round(val,4)