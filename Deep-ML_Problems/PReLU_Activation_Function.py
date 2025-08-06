# Problem 98: Implement the PReLU Activation Function
# PReLU(x)=αx
#-------------------------------------------------------

def prelu(x: float, alpha: float = 0.25) -> float:

	if x > 0:
		return float(x)
	
	elif x ==0:
		return float(x)

	else:
		return float(alpha * x)