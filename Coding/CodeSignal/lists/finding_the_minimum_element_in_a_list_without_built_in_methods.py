"""
You are given a list of integers. Your task is to write a function find_min(nums), that returns the minimum number from the list without using Python's built-in min() function.

If the list is empty, your function should return None.
"""
def find_min(nums):
    if len(nums) == 0:
        return None

    min_val = nums[0]
    for i in range(len(nums)):
        if nums[i] < min_val:
            min_val = nums[i]

    return min_val
