# 238. Product of Array Except Self
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        # one zero, all zero, except itself (itself = mult of remaining values)
        # two zeros, all zeros

        left_val = 1
        right_val = 1
        pre_sum = [0] * len(nums)
        post_sum = [0] * len(nums)

        for i in range(len(nums)):
            j = -i - 1
            pre_sum[i] = left_val
            post_sum[j] = right_val

            left_val *= nums[i]
            right_val *= nums[j]
        
        out_nums = []
        for i, j in zip(pre_sum, post_sum):
            out_nums.append(i * j)
        return out_nums
