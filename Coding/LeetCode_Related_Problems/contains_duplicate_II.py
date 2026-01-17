# 219. Contains Duplicate II
# Given an integer array nums and an integer k, return true if 
# there are two distinct indices i and j in the array such that nums[i] == nums[j] and abs(i - j) <= k.

class Solution:
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        hashm = {}

        for i in range(len(nums)):
            if (nums[i] in hashm) and (abs(i - hashm[nums[i]]) <= k):
                return True
            hashm[nums[i]] = i
        return False