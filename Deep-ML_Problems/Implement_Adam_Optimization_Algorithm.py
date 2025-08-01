# Implement Adam Optimization Algorithm: Problem 49
# Implement the Adam (Adaptive Moment Estimation) optimization algorithm in Python. Adam is an optimization algorithm 
# that adapts the learning rate for each parameter. Your task is to write a function adam_optimizer that updates the 
# parameters of a given function using the Adam algorithm.

#---------------------------------------------------------------------------------------------------------------------
#PyTorch Solution
import torch 

def objective_function(x):
    return x[0]**2 + x[1]**2

def gradient(x):
    return 2 * x


def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10):
    x = x0.clone()
    m = torch.zeros_like(x)
    v = torch.zeros_like(x)  
    
    for i in range(num_iterations):
        t = i + 1
        g_t = grad(x)
        m = beta1 * m + (1-beta1) * g_t
        v = beta2 * v + (1-beta2) * (g_t ** 2)
        m_bt = m /(1-beta1 ** t)
        v_bt = v /(1-beta2 ** t)
        
        x = x - learning_rate * (m_bt / (torch.sqrt(v_bt) + epsilon))
     
    return x 






