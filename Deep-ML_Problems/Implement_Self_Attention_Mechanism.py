#Problem 53 *Implement Self-Attention Mechanism*: Task: Implement the Self-Attention Mechanism
#Your task is to implement the self-attention mechanism, which is a fundamental component of transformer models, 
#widely used in natural language processing and computer vision tasks. The self-attention mechanism allows a model to 
#dynamically focus on different parts of the input sequence when generating a contextualized representation.
#Your function should return the self-attention output as a numpy array.

#----------------------------------------------------------------------------------------------------------------------
#Pytorch Solution
import torch
import torch.nn.functional as F

def compute_qkv(X, w_q, w_k, w_v):
    Q = torch.matmul(X, w_q)
    K = torch.matmul(X, w_k)
    V = torch.matmul(X, w_v)
    return Q, K, V


def self_attention(Q: torch.Tensor, K: torch.Tensor, V:torch.Tensor):
    d = K.size(-1)
    x = torch.matmul(Q, K.T) / torch.sqrt(torch.tensor(d))
    x_s = F.softmax(x, dim=-1) 
    attn = torch.matmul(x_s, V)
    return attn

#----------------------------------------------------------------------------------------------------------------------
#Numpy Solution
"""
import numpy as np

def compute_qkv(X, W_q, W_k, W_v):
    Q = np.matmul(X, W_q)
    K = np.matmul(X, W_k)
    V = np.matmul(X, W_v)

    return Q, K, V


def self_attention(Q:np.ndarray, K:np.ndarray, V:np.ndarray):

    k = np.size(K, -1)

    val = np.matmul(Q, K.T) / np.sqrt(k)
    exp_val = np.exp(val - np.max(val, axis=-1, keepdims=True)) 
    #incase large values in the matrix, we take the difference between the value and the largest 
    # value in the given row and then apply exponential

    scores = exp_val / np.sum(exp_val, axis=-1, keepdims=True)

    attention_output = np.matmul(scores, V)
      
	return attention_output
"""