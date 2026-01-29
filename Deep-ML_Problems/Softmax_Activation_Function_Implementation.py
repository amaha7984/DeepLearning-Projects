# Problem 23: Softmax Activation Function Implementation
# -------------------------------------------------------

import numpy as np
def softmax(scores: list[float]) -> list[float]:
	# e^x/sum(all (e^x))
	scores = np.array(scores)
	exp_scores = np.exp(scores)
	return exp_scores/np.sum(exp_scores)
