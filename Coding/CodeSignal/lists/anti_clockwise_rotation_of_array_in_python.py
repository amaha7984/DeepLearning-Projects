"""
You are provided with an array of n integers and a number k. Your task is to perform an anti-clockwise rotation (toward the front) of the array by k positions. 
The rotation should be done in place, which means you have to directly manipulate the input array without creating a new one. Note that k might be bigger than the array length.

For example, if the input array is [1, 2, 3, 4, 5, 6, 7], and k = 3, then after the operation, the input array should look like [4, 5, 6, 7, 1, 2, 3].
"""
from typing import List

def anti_rotate_array(nums: List[int], k: int) -> None:
    n = len(nums)
    if n == 0:
        return
    
    k %= n 

    def reverse(l, r):
        while l < r:
            nums[l], nums[r] = nums[r], nums[l]
            l += 1
            r -= 1

    reverse(0, k - 1)
    reverse(k, n - 1)
    reverse(0, n - 1)