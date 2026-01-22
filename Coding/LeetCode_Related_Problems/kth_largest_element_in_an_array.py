#215. Kth Largest Element in an Array
import heapq
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:

        for i in range(len(nums)):
            nums[i] = -nums[i]
        heapq.heapify(nums)

        for _ in range(k-1):
            heapq.heappop(nums)
        
        return -heapq.heappop(nums)

#-----------Another Solution-------------#
import heapq
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:

        min_heap = []
        for i in nums:
            if len(min_heap) < k:
                heapq.heappush(min_heap, i)
            else:
                heapq.heappushpop(min_heap, i)
        return min_heap[0]
        