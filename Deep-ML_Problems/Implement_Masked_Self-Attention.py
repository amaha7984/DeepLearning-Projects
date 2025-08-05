# Problem: 107, Implement Masked Self-Attention
# Implement masked self-attention, a variation of the attention mechanism used in sequence modeling tasks such as text generation. 
# Your task is to compute masked self-attention using query (Q), key (K), value (V) matrices and an attention mask.

#------------------------------------------------------------------------------------------------------------------
#Numpy Solution

import numpy as np

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
	
	return np.dot(X, W_q), np.dot(X, W_k), np.dot(X, W_v)

def softmax(x: np.ndarray):
    exp_val = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_val / np.sum(exp_val, axis=-1, keepdims = True)

def masked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray) -> np.ndarray:
	
    k = np.size(K, -1)

    scores = np.matmul(Q, K.T) / np.sqrt(k)

    masked_score = softmax(scores + mask)

    masked_attn = np.matmul(masked_score, V)
    return masked_attn