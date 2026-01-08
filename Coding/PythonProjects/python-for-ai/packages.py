# import packages patters
# Two ways
# Pattern 1: import whole module
import math
# we can use: math.sqrt(16)

from math import sqrt, pi
# we can use: sqrt(16)

#----import with alias------#
import pandas as pd 
df = pd.DataFrame(data)

################################
# Creating requirements.txt
# List all project’s packages:

pip freeze > requirements.txt
