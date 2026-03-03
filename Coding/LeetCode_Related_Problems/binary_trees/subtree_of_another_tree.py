# 572. Subtree of Another Tree

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
from collections import deque
class Solution:
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:

        def sameTree(node1, node2):
            if not node1 and not node2:
                return True
            if not node1 or not node2:
                return False
            if node1.val != node2.val:
                return False
            return (sameTree(node1.left, node2.left) and
                   sameTree(node1.right, node2.right))

        q = deque()
        q.append(root)
        
        while q:
            node = q.popleft()
            if node.val == subRoot.val:
                if sameTree(node, subRoot):
                    return True

            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        
       
        return False
    
# Time Complexity: Let K = count of nodes in root with value == subRoot.val
# Time is O(N + K·M), where N is the total nodes to traverse over the main tree "root", and M is the total nodes in "subRoot" tree

            