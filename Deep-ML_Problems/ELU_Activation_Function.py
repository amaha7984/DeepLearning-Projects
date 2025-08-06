# Problem 97: Implement the ELU Activation Function
# the ELU activation is computed as alpha*(np.exp(x) - 1)

#---------------------------------------------------------

import numpy as np

def elu(x: float, alpha: float = 1.0) -> float:
    if x > 0:
        return float(x)
    
    elif x==0:
        return float(x)
    
    else:
        return np.round(alpha * (np.exp(x) - 1), 4)
	

	