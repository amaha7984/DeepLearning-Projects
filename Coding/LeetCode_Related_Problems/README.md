## PyTorch Quick Notes

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

### Hash Set
- Hash Set is a data structure that stores unique elements in an unordered manner
- Hashing ensures that the set operations like add, remove, and lookup can be done at a constant time O(1)
- Initialize with 'set()'
- Example of Creating Hash set: 
  - hs = {1, 2, 3, 4, 5}
  - hs1 = set([1, 2, 3, 3, 4])