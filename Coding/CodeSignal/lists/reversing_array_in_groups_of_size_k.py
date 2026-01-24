"""
You have been given an array of n integers. Your task is to write a function that reverses the array in groups of k size, and if the last group has fewer than k elements, reverse all of them. 
Return the newly organized array after the groups have been reversed.
For example, given the array [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] and k = 3, the output should be: [3, 2, 1, 6, 5, 4, 9, 8, 7, 10].
The first three elements are reversed to get [3, 2, 1], the next three become [6, 5, 4], the following three are [9, 8, 7], and the final one remains [10] as there are fewer than k elements remaining.
"""
def solution(numbers, k):
    n = len(numbers)
    i = 0
    
    while i < n:
        
        current = min(i + k -1, n -1) # End this group either at size k, or at the last element — whichever comes first.
        l = i
        r = current
        while l < r:
            numbers[l], numbers[r] = numbers[r], numbers[l]
            l += 1
            r -= 1
        i += k
    return numbers
    