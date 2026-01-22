"""
You are given a list of integers. Your task is to write a Python function to find the second-largest number among these integers. If the list has fewer than two unique numbers, return None.

You are not allowed to use any built-in Python functions or methods such as sorted(), max(), or sort(). Instead, you should implement the task using basic list operations.
"""

def second_max(nums: List[int]) -> Optional[int]:
    largest = None
    sec_largest = None
    
    for i in range(len(nums)):
        if largest is None or nums[i] > largest:
            sec_largest = largest
            largest = nums[i]
        elif nums[i] < largest and (sec_largest is None or nums[i] > sec_largest):
            sec_largest = nums[i]
 
    return sec_largest
