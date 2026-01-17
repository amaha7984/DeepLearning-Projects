class Solution:
    def twoSum(self, nums, target):
        hash = {}

        for i in range(len(nums)):
            hash[nums[i]] = i

        for j in range(len(nums)):
            value = target - nums[i]

            if value in hash and hash[value] != j:
                return [j, hash[value]]

#---Another solution
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:

        hashdict = {}
        for i in range(len(nums)):
            if target - nums[i] in hashdict:
                return [i, hashdict[target - nums[i]]]
            
            hashdict[nums[i]] = i