# 26. Remove Duplicates from Sorted Array

class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return 1
        
        i = 1
        j = 1

        while i < len(nums):
            if nums[i] != nums[i - 1]:
                nums[j] = nums[i]
                j += 1
            i += 1
        
        return j

                


        
