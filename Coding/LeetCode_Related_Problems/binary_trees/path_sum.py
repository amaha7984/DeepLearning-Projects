# 112. Path Sum

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        if not root:
            return False
        
        stk = [(root, root.val)]
        
        while stk:
            node, total = stk.pop()

            if node.right:
                stk.append((node.right, total + node.right.val))
            if node.left:
                stk.append((node.left, total + node.left.val))

            if not node.left and not node.right:
                if total == targetSum:
                    return True
        
        return False

        