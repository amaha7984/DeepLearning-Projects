"""
You are given a list of n integers and a number k. Your task is to shuffle the array in such a way that, starting from the first element, 
every k-th element moves to the end of the array.For instance, if nums = [1, 2, 3, 4, 5, 6, 7, 8] and k = 3, the output should be [1, 2, 4, 5, 7, 8, 3, 6]. 
Here, the 3rd element 3 and the 6th element 6 (every 3rd element starting from the first) are moved to the end of the array.
"""
def shuffle_array(nums, k):
    n = len(nums)
    if n == 0 or k <= 1:
        return nums

    moved = 0            # how many elements already moved to the end
    i = k - 1            # first k-th (0-based)

    while i < n - moved:
        nums.append(nums.pop(i))   # move k-th to end (keeps order)
        moved += 1
        i += (k - 1)               # next k-th in the remaining prefix

    return nums
    