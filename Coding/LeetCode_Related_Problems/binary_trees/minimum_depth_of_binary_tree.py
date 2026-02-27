# 111. Minimum Depth of Binary Tree

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def minDepth(self, root: Optional[TreeNode]) -> int:
        if root is None:
            return 0
        
        if root.left is None:
            return 1 + self.minDepth(root.right)
        
        if root.right is None:
            return 1 + self.minDepth(root.left)

        return 1 + min(self.minDepth(root.left), self.minDepth(root.right))

#-----------------------BFS Solution--------------------------------------#
from collections import deque
class Solution:
    def minDepth(self, root: Optional[TreeNode]) -> int:

        if not root:
            return 0

        q = deque()
        q.append((root, 1))

        while q:
            node, min_depth = q.popleft()

            if not node.left and not node.right:
                return min_depth
            
            if node.left:
                q.append((node.left, min_depth + 1))
            
            if node.right:
                q.append((node.right, min_depth + 1))

        
            