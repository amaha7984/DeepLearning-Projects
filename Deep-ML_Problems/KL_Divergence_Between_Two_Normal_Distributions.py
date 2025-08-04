# Problem 56 : KL Divergence Between Two Normal Distributions
# Task: Implement KL Divergence Between Two Normal Distributions
# -----------------------------------------------------------------------------
#PyTorch Solution
import torch
def kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q):

	result = torch.log(torch.tensor(sigma_q / sigma_p)) + ((sigma_p**2 + (mu_p - mu_q)**2)/ (2 * sigma_q**2)) - 1/2
 
	return result




# -----------------------------------------------------------------------------
# Numpy Solution
"""
import numpy as np

def kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q):

	result = np.log(sigma_q / sigma_p) + ((sigma_p**2 + (mu_p - mu_q)**2)/ (2 * sigma_q**2)) - 1/2
 
	return result
"""