# 378. Kth Smallest Element in a Sorted Matrix

#------------This will be 0(n^2)----need to find another solution---------------#
class Solution:
    def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
        flat = []
        for l in matrix:
            flat.extend(l)
        
        flat.sort()
        return flat[k-1]

        