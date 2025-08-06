# Problem 100: Implement the Softsign Activation Function

def softsign(x: float) -> float:
	if x == 0:
		return 0.0
	else:
		val = x / ( 1 + abs(x))
		return round(val,4)