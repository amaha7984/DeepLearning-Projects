# Problem 151: Dropout Layer 
# Implement a dropout layer that applies random neuron deactivation during training to prevent overfitting in neural networks. 

# ---------------------------------------------------------------------------------------------------------------------------
#Numpy Solution

import numpy as np

class DropoutLayer:
    def __init__(self, p: float):
        self.p = p
        self.keep = 1.0 - p
        self.mask = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if not training or self.p == 0.0:
            
            return x
     
        self.mask = np.random.binomial(1, self.keep, size=x.shape).astype(x.dtype)
        return (x * self.mask) / self.keep

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return (grad * self.mask) / self.keep
