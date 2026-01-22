"""
You are given an array of integers. Your job is to return the count of even and odd integers in the given array without using any built-in Python methods.

Your function should return a tuple in the format (even_count, odd_count), where even_count represents the number of even integers and odd_count represents the number of odd integers in the provided array.
"""

def solution(nums):
    odd_n = 0
    even_n = 0
    
    for i in range(len(nums)):
        if nums[i] % 2 == 0:
            even_n += 1
        else:
            odd_n += 1
    return (even_n, odd_n)