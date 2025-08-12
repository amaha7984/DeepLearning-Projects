# Problem 126: Write a Python function to perform Group Normalization 
#on a 4D input tensor with shape (B, C, H, W). The function should normalize over 
#smaller groups of channels, then apply a learned scale (gamma) and shift (beta).

#---------------------------------------------------------------------------------
#numpy solution

import numpy as np

def group_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, num_groups: int, epsilon: float = 1e-5) -> np.ndarray:

    X_newshape = X.shape[1] // num_groups
    X_new = X.reshape(X.shape[0], num_groups, X_newshape, X.shape[2], X.shape[3])
    mean = np.mean(X_new, axis=(2,3,4), keepdims = True)
    var = np.var(X_new, axis=(2,3,4), keepdims = True)

    X_norm = (X_new - mean) / (np.sqrt(var + epsilon))

    X_res = X_norm.reshape(X.shape[0], X.shape[1], X.shape[2], X.shape[3])
    C = X.shape[1]

    gamma = gamma.reshape(1, C, 1, 1)
    beta = beta.reshape(1, C, 1, 1)

    return (X_res * gamma) + beta 
