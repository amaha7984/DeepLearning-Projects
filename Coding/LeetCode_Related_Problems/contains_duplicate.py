# O(n^2) time
# 0(1) space

class Solution:
    def hasDuplicate(self, nums: List[int]) -> bool:
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                if nums[j] == nums[i]:
                    return True
        
        return False
    
#------------------------------------------------------#
# O(n) time complexity
# O(n) space complexity

class Solution:
    def hasDuplicate(self, nums: List[int]) -> bool:

        hashs = set()
        for i in range(len(nums)):
            if nums[i] not in hashs:
                hashs.add(nums[i])
            elif nums[i] in hashs:
                return True
        return False