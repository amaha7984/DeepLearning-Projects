"""
You are given an array of n integers. Your task is to reverse the array without using any additional lists or the built-in reversed() function.

Amend the array in-place and return the array. In-place here means you are not allowed to use any additional lists in your solution.
"""
def solution(arr):
    i = 0
    j = len(arr) - 1
    while i < j:
        arr[i], arr[j] = arr[j], arr[i]
        i += 1
        j -= 1
    return arr