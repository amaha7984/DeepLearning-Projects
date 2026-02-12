# This folder contains the information about coding in Python
# Credits: https://python.datalumina.com/basics
# Thanks to Dave Ebbelaar


## Data types
### Strings
- We can use either use double quotation mark as: string = "My name is Arpan"
- Or we can use triple quotation mark: string = """My name is Arpan.
                                                I live in Miami."""
- Note with triple quotation mark, if we have line break then it will also appear same in the printed statement.

#### Combining Strings
See file "strings.py"

#### String length: len() keyword can be used to print the number of characters in string
See file "strings.py"

#### Converting to string: str() can be used to convert other types to strings
See file "strings.py" 

#### f-Strings 
See file "strings.py"

### Booleans: True and False Values for decisions

#### Creating Booleans
- Direct Assignment
  - is_generated = True
- From Comparisons
  - image_size = 256
  - valid_size = image_size == 256


## Control Flow
### If statements
See file "if_else_statements.py"

### Loops
#### For Loops
- Python start index is 0. This is called “zero-indexing”
- See an example file "loops.py"

## Data Structures
### Lists
See "lists.py" for all loops' functionalities

### Dictionaries
- Store data with key-value pairs
- See dictionaries.py for detailed information on dictionaries methods and usage

### Tuples
- Similar to lists and can be instantiated with ()
- Tuples are immutable

### Sets
- Sets are collections that only store unique values. They automatically remove duplicates.
- Sets can be created with set() or with curly braces {}
- Important note: use curly braces to create a set when it has values

## Functions
#### Defining Functions
- Use def keyword to define function
- Use lowercase letters, separate words with underscores, and be descriptive about what it does
#### Parameters
- We can pass data into functions with parameters
- See functions.py for detailed information on Functions

#### Return Values
- Get results back from functions 
- See functions.py for more information

## Packages 
- There are two types:
  - Built-in: Come with Python (no installation needed)
  - External: Need to install first with pip

- Few Terminologies:
  - Module: A single Python file (like math.py)
  - Package: A folder containing multiple modules
  - Function: A reusage block of code
  - Class: A blueprint of creating objects

### Where to find packages
- [PyPI](https://pypi.org/)
- [awesome-python](https://github.com/vinta/awesome-python)

## Sorting in Python

### Sorting a List
- For a given list, nums:
- If we want to sort the list and create a copy of the list, we can do: nums = sorted(nums)
- If we want to modify the list by sorting it, we can do: nums.sort(). This is in place sorting.
    

















