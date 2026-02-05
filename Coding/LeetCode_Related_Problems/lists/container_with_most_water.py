# 11. Container With Most Water

class Solution:
    def maxArea(self, height: List[int]) -> int:
        l, r = 0, len(height) - 1
        max_amount = 0

        while l < r:
            width = r - l
            max_amount = max(max_amount, (min(height[l], height[r])) * width)

            if height[l] < height[r]:
                l += 1
            
            else:
                r -= 1

        return max_amount

