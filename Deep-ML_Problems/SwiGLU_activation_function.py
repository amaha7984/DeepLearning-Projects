# Problem 156: Implement SwiGLU activation function
# ----------------------------------------------------

#Numpy Solution

import numpy as np

def SwiGLU(x: np.ndarray) -> np.ndarray:
    """
    Args:
        x: np.ndarray of shape (batch_size, 2d)
    Returns:
        np.ndarray of shape (batch_size, d)
    """
    d = x.shape[1]//2
    x1 = x[:, :d]
    x2 = x[:, d:]

    silu_x2 = np.multiply(x2, 1/(1+np.exp(-x2))) 
    result = np.multiply(x1, silu_x2)

    final = []
    for i in result:
        final.append(np.round(i, 4))
    
    return final


