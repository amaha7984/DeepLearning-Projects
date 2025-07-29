#Implement Multi-Head Attention: Question 94
#Implement the multi-head attention mechanism, a critical component of transformer models.
#Given Query (Q), Key (K), and Value (V) matrices, compute the attention outputs for multiple heads and concatenate the results.

#---------------------------------------------------------------------------------------------------------------------------------
#PyTorch Solution


import torch
import torch.nn.functional as F

def compute_qkv(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor):

    Q = torch.matmul(X, W_q)
    K = torch.matmul(X, W_k)
    V = torch.matmul(X, W_v)

    return Q, K, V

def self_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):

    k = K.size(-1)

    val = torch.matmul(Q, K.T)/torch.sqrt(k)

    scores = torch.softmax(val, dim=1) #row-wise softmax

    attn = torch.matmul(scores, V)
    return attn

def multi_head_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, n_heads: int):

    dim = Q.size(-1)

    d_dim = dim // n_heads

    heads_attn = []

    for i in range(n_heads):

        Q_i = Q[:, i * d_dim:(i+1) * d_dim]
        K_i = K[:, i * d_dim:(i + 1) * d_dim]
        V_i = V[:, i * d_dim:(i + 1) * d_dim]


        mult_attn = self_attention(Q_i, K_i, V_i)
        heads_attn.append(mult_attn)

    result = torch.cat(heads_attn, dim=-1)

    return result
    
#-----------------------------------------------------------------------------------------------------
#numpy solution

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
    scores = exp_val / np.sum(exp_val, axis=-1, keepdims=True)

    attention_output = np.matmul(scores, V)
      
	return attention_output

def multi_head_attention(Q:np.ndarray, K:np.ndarray, V:np.ndarray, n_heads: int):
	dim = np.size(Q, -1)
    d_dim = dim // n_heads

    heads_attn = []

    for i in range(n_heads):

        Q_i = Q[:, i * d_dim:(i+1) * d_dim]
        K_i = K[:, i * d_dim:(i + 1) * d_dim]
        V_i = V[:, i * d_dim:(i + 1) * d_dim]

        mult_attn = self_attention(Q_i, K_i, V_i)
        heads_attn.append(mult_attn)
    
    result = np.concatenate(heads_attn, axis=1)

    return result
"""

    


    

    