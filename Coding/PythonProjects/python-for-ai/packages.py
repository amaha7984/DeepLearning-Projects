# import packages patters
# Two ways
# Pattern 1: import whole module
"""
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
"""
#################################
# External packages
# Web requests
import requests

response = requests.get("https://api.example.com/data")
data = response.json()

# Data Analysis
import pandas as pd

data = {'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['NYC', 'LA', 'Chicago']
    }

df = pd.DataFrame(data)
print(df)