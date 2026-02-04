# 4. Median of Two Sorted Arrays
"""
Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.

The overall run time complexity should be O(log (m+n)).
"""

class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        # MergeSort to attach two arrays, nums_out
        # nums_out %2 ==0, [1,2,3,4,5], middle = n // 2, median = (nums_out[middle-1] + nums_out[middle])/2
        # nums_out %2 > 0, middle = n // 2, median = nums_out[middle]
        m, n = len(nums1), len(nums2)

        nums_out = [0] * (m + n)

        l, r = 0, 0
        i = 0

        while l < m and r < n:
            if nums1[l] < nums2[r]:
                nums_out[i] = nums1[l]
                l += 1
            else:
                nums_out[i] = nums2[r]
                r += 1
            i += 1
        
        while l < m:
            nums_out[i] = nums1[l]
            l += 1
            i += 1
        
        while r < n:
            nums_out[i] = nums2[r]
            r += 1
            i += 1
         
        if len(nums_out) % 2 == 0:
            middle = len(nums_out) // 2
            return (nums_out[middle - 1] + nums_out[middle]) / 2
        else:
            middle = len(nums_out) // 2
            return nums_out[middle]
        