# Empty 
my_dict = {}

#General way to create dictionaries
animal = {
    "category": "lion",
    "legs": 4,
    5: 9,
    "weight": 240
}


# print(animal[5])
#print(animal.get("category"))

#Another way to create dictionaries
person = dict(name="Arpan", age=22)

# print(person["name"])

##########################################
# Dictionaries methods
person = {"name": "Alice", "age": 30, "city": "New York"}

# Get all keys, values, or items
print(person.keys())   #prints: dict_keys(['name', 'age', 'city'])
print(person.values()) #prints: dict_values(['Alice', 30, 'New York'])
print(person.items())  #prints: dict_items([('name', 'Alice'), ('age', 30), ('city', 'New York')])

##########################################
# Nested dictionaries
person = {
    "Arpan": {"age": 28, "grade": "A"},
    "bob": {"age": 29, "grade": "A"}
}

print(person["Arpan"]["age"])
