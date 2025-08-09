# Problem 115: Implement Batch Normalization for BCHW Input

import numpy as np

def batch_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    mean = np.mean(X, axis=(0,2,3), keepdims = True)
    variance = np.var(X, axis=(0,2,3), keepdims = True)

    C = X.shape[1]
    
    X_norm = (X-mean) / np.sqrt(variance + epsilon)
    
    out = gamma.reshape(1, C, 1, 1) * X_norm + beta.reshape(1, C, 1, 1)

    return out
