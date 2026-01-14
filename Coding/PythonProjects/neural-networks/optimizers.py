import numpy as np

#-------------------SGD--------------------------------#
# weight = weight - learning_rate * gradient_of_loss_wrt_weight

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, params, grads):
        # params, grads: lists of numpy arrays
        for p, g in zip(params, grads):
            p -= self.lr * g

#--------------------Momentum SGD----------------------#
# velocity = momentum * previous_velocity + gradient_of_loss_wrt_weight
# weight   = weight - learning_rate * velocity
class SGD:
    def __init__(self, lr=0.01, momentum=0.0):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def step(self, params, grads):
        if self.v is None:
            self.v = [np.zeros_like(p) for p in params]

        for i, (p, g) in enumerate(zip(params, grads)):
            self.v[i] = self.momentum * self.v[i] + g
            p -= self.lr * self.v[i]


#------------------------Adam--------------------------#
# Adam combines momentum with per-parameter adaptive learning rates by tracking the first and second moments of gradients.
# m = exponential moving average of gradients (momentum term)
# v = exponential moving average of squared gradients (adaptive scaling)
# m_hat = bias-corrected first moment
# v_hat = bias-corrected second moment
# weight = weight - learning_rate * m_hat / (sqrt(v_hat) + epsilon)

class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]

        self.t += 1

        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
