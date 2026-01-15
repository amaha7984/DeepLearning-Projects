class Solution:
    def twoSum(self, nums, target):
        hash = {}

        for i in range(len(nums)):
            hash[nums[i]] = i

        for j in range(len(nums)):
            value = target - nums[i]

            if value in hash and hash[value] != j:
                return [j, hash[value]]