# 617. Merge Two Binary Trees

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def mergeTrees(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> Optional[TreeNode]:

        if not root1 and not root2:
            return None 

        out_tree = TreeNode(0)
        
        if root1 and not root2:
            out_tree.val = root1.val
            out_tree.left = self.mergeTrees(root1.left, None)
            out_tree.right = self.mergeTrees(root1.right, None)

        elif not root1 and root2:
            out_tree.val = root2.val
            out_tree.left = self.mergeTrees(root2.left, None)
            out_tree.right = self.mergeTrees(root2.right, None)

        else:
            out_tree.val = root1.val + root2.val
            out_tree.left = self.mergeTrees(root1.left, root2.left)
            out_tree.right = self.mergeTrees(root1.right, root2.right)

        return out_tree


            