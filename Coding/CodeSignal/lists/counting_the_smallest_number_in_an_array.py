"""
You are given an array of integers. Your task is to write a function in Python that returns the number of times the smallest element appears in the array.

Please note that built-in methods such as min() or count() should not be used in this task. Your goal is to accomplish this task by iterating over the array elements manually. 
Try to solve the task by doing just a single list traversal.
"""
def count_min(numbers):
    if len(numbers) == 0:
        return 0
    min_value = numbers[0]
    count = 1
    for i in range(1, len(numbers)):
        if numbers[i] < min_value:
            min_value = numbers[i]
            count = 1
        elif numbers[i] == min_value:
            count += 1
    return count
            
            