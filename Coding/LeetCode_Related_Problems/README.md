### Hashmaps
- Dictionaries in Python
- Allow search in O(1)
- key in hashmap is immutable; once a value assigned to a key, key is unchangeable
- Initialize with {} or dict()
- We need to initialize each key, to avoid it, we can use defaultdict() to first initialize hashmap then no need to initialize each key
#### Retrieving Data
- hashmap.keys()
- hashmap.values()
- hashmap.items()
#### Sorting Hashmaps with keys or values
- Sorting in descending order with keys, let's say h_map is a hashmap or dictionary
  - sorted(h_map.items(), key=lambda x: x[0], reverse=True) 
- Sorting in descending order with values, let's say h_map is a hashmap or dictionary
  - sorted(h_map.items(), key=lambda x: x[1], reverse=True) 

### Hash Set
- Hash Set is a data structure that stores unique elements in an unordered manner
- Hashing ensures that the set operations like add, remove, and lookup can be done at a constant time O(1)
- Initialize with 'set()'
- Example of Creating Hash set: 
  - hs = {1, 2, 3, 4, 5}
  - hs1 = set([1, 2, 3, 3, 4])
- Basic Operation:
  - let's say a hasset, hset = set([1, 2, 3, 3, 4])
  - Addition with 'hset.add(5)'
  - Removable with 'hset.remove(2)'
- We can look up item in hash set in constant time.

### Similarities with String and a List:
- In Python, strings and lists support many of the same operations (indexing, len(), loops, slicing).

- Indexing works the same:
  - s = "609" → s[0] == '6'
  - lst = [6, 0, 9 ] → lst[0] == 6

- Length works the same: len(s) and len(lst)

- Looping works the same:
  - for c in s: ...
  - for x in lst: ...

- range(len(...)) + indexing works the same:
  - for i in range(len(s)): s[i]
  - for i in range(len(lst)): lst[i]

- Slicing works similarly:
  - s[1:3] returns a string
  - lst[1:3] returns a list

- Key difference: strings are immutable (read-only):
  - s[0] = '9' → ❌ error

- Lists are mutable (can be modified):
  - lst[0] = '9' → ✅ works

- Note: a string is a read-only list of characters
- Rule of thumb:
  - If we need to calculate, use numbers
  - If we need to compare, match, or transform digits, use strings

### Alphanumeric characters
- Chacters inclusive from A - Z, a - z, 0 - 9
- In python, we can access the Ascii value of any characters with ord(c), where c is any character
- To determine if any character is alphanumeric, we can do return the boolean value of: 
- (ord('A') <= ord(c) <= ord('Z') or ord('a') <= ord(c) <= ord('z') or ord('0') <= ord(c) <= ord('9'))

