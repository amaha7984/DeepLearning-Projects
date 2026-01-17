class Solution:
    def subarraySum(self, nums, k):

        hash_arr = {}
        ans = 0
        sum = 0

        for i in range(len(nums)):
            sum += nums[i]

            if (sum == k):
                ans += 1
            
            if (sum - k in hash_arr):
                ans += hash[sum - k]
            
            if sum in hash_arr:
                hash_arr[sum] += 1
            else:
                hash_arr[sum] = 1
        return ans



