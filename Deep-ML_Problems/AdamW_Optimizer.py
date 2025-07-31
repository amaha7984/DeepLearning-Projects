#AdamW Optimizer
# Implement AdamW similar to the Adam

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#PyTorch Solution
import torch 

def round_tensor(x): 
    return torch.round(x * 1e5) / 1e5


def adamW_optimizer(parameter: torch.Tensor, grad: torch.tensor, m: torch.Tensor, v: torch.Tensor, t: int, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, decay = 0.001):

    grad = torch.tensor(grad, dtype=torch.float32) #for safe: grad = torch.as_tensor(grad, dtype=torch.float32) 

    m_t = beta1 * m + (1-beta1) * grad
    v_t = beta2 * v + (1-beta2) * grad**2

    m_bt = m_t/(1-beta1**t)
    v_bt = v_t/(1-beta2**t)
    
    parameter = parameter - learning_rate * ((m_bt / ((torch.sqrt(v_bt)) + epsilon)) + decay * parameter)
    return round_tensor(parameter), round_tensor(m_t), round_tensor(v_t)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Numpy Solution

"""
import numpy as np

def adam_optimizer(parameter, grad, m, v, t, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, decay = 0.001):
    m_t = beta1 * m + (1-beta1) * grad
    v_t = beta2 * v + (1-beta2) * grad**2

    m_bt = m_t/(1-beta1**t)
    v_bt = v_t/(1-beta2**t)

    parameter = parameter - learning_rate * ((m_bt / (np.sqrt(v_bt) + epsilon)) + decay * parameter)
    return np.round(parameter,5), np.round(m_t,5), np.round(v_t,5)

"""