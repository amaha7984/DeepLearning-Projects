# Takes O(n) time
class NumArray:

    def __init__(self, nums: List[int]):
        self.nums = nums
        

    def sumRange(self, left: int, right: int) -> int:
        sum = 0
        for i in range(left, right+1):
            sum += self.nums[i]
        return sum
    
#---------------------------------------------------------#
# use prefix sum inside constructor with O(n) time
# compute the sum query with O(1) time in function
class NumArray:

    def __init__(self, nums: List[int]):
        self.pre_sum = []
        sum = 0
        for i in range(len(nums)):
            sum += nums[i]
            self.pre_sum.append(sum)

    def sumRange(self, left: int, right: int) -> int:
        right_sum = self.pre_sum[right]
        if left > 0:
            left_sum = self.pre_sum[left - 1]
        else:
            left_sum = 0
        return (right_sum - left_sum)
        