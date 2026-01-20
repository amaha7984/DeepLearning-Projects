# 347. Top K Frequent Elements
# Given an integer array nums and an integer k, return the k most frequent elements. You may return the answer in any order.

class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        fre_num = {}
        
        for i in range(len(nums)):
            if nums[i] not in fre_num:
                fre_num[nums[i]] = 1
            elif nums[i] in fre_num:
                fre_num[nums[i]] += 1
        sorted_items = sorted(fre_num.items(), key=lambda x: x[1], reverse=True)        
        k_fre = []
        #sort the hashmap with values in descending order
        n = 0
        for key, value in sorted_items:
            if n < k:
                k_fre.append(key)
                n += 1
        return k_fre
