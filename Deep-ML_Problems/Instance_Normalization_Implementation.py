# Problem 143: Instance Normalization (IN) Implementation
# Implement the Instance Normalization operation for 4D tensors (B, C, H, W) using NumPy.

# -----------------------------------------------------------------------------------------
#Numpy Solution

import numpy as np

def instance_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """
    Perform Instance Normalization over a 4D tensor X of shape (B, C, H, W).
    gamma: scale parameter of shape (C,)
    beta: shift parameter of shape (C,)
    epsilon: small value for numerical stability
    Returns: normalized array of same shape as X
    """
    mean = np.mean(X, axis=(2,3), keepdims=True)
    var = np.var(X, axis=(2,3), keepdims=True)

    X_norm = (X - mean) / (np.sqrt(var+epsilon))

    C = X.shape[1]

    gamma = np.reshape(gamma, (1,C,1,1))
    beta = np.reshape(beta, (1,C,1,1))

    X_ins = gamma * X_norm + beta



    return X_ins