first_name = "Arpan"
last_name = "Mahara"

#concatentation
"""full_name = first_name + " " + last_name
print(full_name)

#repetetion
what = "?" * 5
print(what)

# string length
#len()
print(len(first_name))

#Converting other types to string
age = 28
print("I am " + str(age) + " years old")

"""

#usage of f-Strings
# Instead of: 
print("My name is " + first_name + " " + last_name)
# We can use:
print(f"My name is {first_name} {last_name}")

# Evaluate Expressions with f-Strings
apples = 10
orange = 5
bananas = 10
print(f"{first_name} has {apples + orange + bananas} fruits in total.")
