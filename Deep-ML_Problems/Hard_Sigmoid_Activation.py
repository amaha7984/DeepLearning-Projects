# Implement the Hard Sigmoid Activation Function: problem 96
# Implement the Hard Sigmoid activation function, a computationally efficient approximation
# of the standard sigmoid function. Your function should take a single input value and return 
# the corresponding output based on the Hard Sigmoid definition.
#--------------------------------------------------------------------------------------------

#plain python solution
def hard_sigmoid(x: float) -> float:
	# HardSigmoid(x)=clip(0.2x+0.5, 0, 1)

    if x <= -2.5:
        return 0.0
    elif x >= 2.5:
        return 1.0
    else:
        return 0.2*x + 0.5