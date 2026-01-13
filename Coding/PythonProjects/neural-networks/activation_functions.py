import numpy as np

class ReLU:
    def forward(self, x):
        self.output = np.maximum(0, x)

class LeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x):
        self.output = np.where(x >= 0, x, self.alpha * x)
        
class Tanh:
    def forward(self, x):
        self.output = (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

