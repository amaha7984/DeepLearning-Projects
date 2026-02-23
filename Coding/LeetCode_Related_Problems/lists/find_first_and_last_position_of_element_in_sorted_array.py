#34. Find First and Last Position of Element in Sorted Array

class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        L = 0
        R = len(nums) - 1

        def searchLeft(x, L, R):

            left_val = -1
            while L <= R:
                m = (L + R) // 2

                if x[m] < target:
                    L = m + 1
                elif x[m] > target:
                    R = m - 1
                else:
                    left_val = m
                    R = m - 1
            return left_val

        def searchRight(x, L, R):
            
            right_val = -1

            while L <= R:
                m = (L + R) // 2

                if x[m] < target:
                    L = m + 1
                elif x[m] > target:
                    R = m - 1

                else:
                    right_val = m
                    L = m + 1
            return right_val
        
        left = searchLeft(nums, L, R)
        if left == -1:
            return [-1, -1]
        
        right = searchRight(nums, L, R)
        return [left, right]

