# Problem 102: Implement the Swish Activation Function
# ----------------------------------------------------

#Numpy Solution

import numpy as np
def swish(x: float) -> float:

    if x == 0:
        return 0.0
    else:
        return x / (1 + np.exp(-x))