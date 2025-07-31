#Adam Optimizer: Problem 87
# Implement the Adam optimizer update step function. Your function should take the current parameter value, gradient, and moving averages as inputs, 
# and return the updated parameter value and new moving averages. The function should also handle scalar and array inputs and include bias correction for the moving averages.

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#PyTorch Solution
import torch 

def round_tensor(x): 
    return torch.round(x * 1e5) / 1e5


def adam_optimizer(parameter: torch.tensor, grad: torch.tensor, m: torch.Tensor, v: torch.Tensor, t: int, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):

    grad = torch.tensor(grad, dtype=torch.float32)
    m_t = beta1 * m + (1-beta1) * grad
    v_t = beta2 * v + (1-beta2) * grad.pow(2)

    m_bt = m_t/(1-beta1**t)
    v_bt = v_t/(1-beta2**t)
    
    parameter = parameter - learning_rate * (m_bt / ((torch.sqrt(v_bt)) + epsilon))
    return round_tensor(parameter), round_tensor(m_t), round_tensor(v_t)
    

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Numpy Solution

"""
import numpy as np

def adam_optimizer(parameter, grad, m, v, t, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    m_t = beta1 * m + (1-beta1) * grad
    v_t = beta2 * v + (1-beta2) * grad**2

    m_bt = m_t/(1-beta1**t)
    v_bt = v_t/(1-beta2**t)

    parameter = parameter - learning_rate * (m_bt / (np.sqrt(v_bt) + epsilon))
    return np.round(parameter,5), np.round(m_t,5), np.round(v_t,5)

"""