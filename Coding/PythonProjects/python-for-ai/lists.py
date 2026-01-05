# Creating an empty list

my_list = []

# List can contain item of different types

my_list_mixed = ["Arpan", 28, True, 5.6]

# List are indexed starting at 0
fruits = ["apple", "banana", "orange"]
print(fruits[0]) # This prints apple

# We can access list item from last with "-1"
print(fruits[-1])   # "orange" (last item)
print(fruits[-2])   # "banana" (second to last)

##################################################
# Lists are mutable 

#Add items
fruits.append("grape")
fruits.insert(1, "pineapple")

#Remove items
fruits.remove("banana")
del fruits[0] # Remove by index

last = fruits.pop()        # Remove and return last

# number of items in the list
print(len(fruits))

##################################################
# List Methods

numbers = [3, 1, 4, 1, 5, 9]
# Sorting
numbers.sort()              # Sort in place
print(numbers)              # [1, 1, 3, 4, 5, 9]

numbers.reverse()           # Reverse order
print(numbers)              # [9, 5, 4, 3, 1, 1]

# Copy
new_list = numbers.copy()   # Create a copy