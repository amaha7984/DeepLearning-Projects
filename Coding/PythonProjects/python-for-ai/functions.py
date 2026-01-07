"""
def name(): #defining function
    print("Arpan Mahara")

name() #Calling function

#Global Variables 
discount_rate = 0.15  # Global variable

def apply_discount(price):
    discount = price * discount_rate  # Can read global variable
    return price - discount

result = apply_discount(100)
print(result) 

#Modifying Global Variable
counter = 0  # Global variable

def increment():
    global counter  # Declare we want to modify the global variable
    counter += 1

increment()
print(counter) 



###########################################################
# Parameters
def introduce(name, age):
    print(f"My name is {name}. I am {age} years old.")

introduce("Arpan Mahara", 28)


# --------Default values ---------#
# Keep optional value first 
def greet(name, greeting="Hello"): # def greet(greeting="Hello", name) won't work
    print(f"{greeting}, {name}!")

# Use default
greet("Alice")           # Hello, Alice!

# Override default
# greet("Bob", "Hi")       # Hi, Bob!


###########################################################
# Keyword arguments
# Call functions using parameter names for clarity:
def create_profile(name, age, city):
    print(f"{name}, {age}, from {city}")

# Positional arguments (order matters)
create_profile("Alice", 25, "NYC")

# Keyword arguments (order doesn't matter)
create_profile(city="NYC", age=25, name="Alice")
"""
###########################################################
#------------return values------------#
def multiplication(a, b):
    return a * b

print(multiplication(2, 3))

# returning multiple values
def get_min_max(numbers):
    return min(numbers), max(numbers) #Note: min and maximum are predefined functions

print(get_min_max([1, 3, 4, 8]))
