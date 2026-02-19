# 235. Lowest Common Ancestor of a Binary Search Tree

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

#---------------Solution 1 -------------------------------------------------------------------#
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        LCA = [root] #creating a list with root as first value and global variable
        
        def search(root):
            if not root:
                return
            
            LCA[0] = root

            if root is p or root is q:
                return 
            
            elif root.val > p.val and root.val > q.val:
                search(root.left)
            elif root.val < p.val and root.val < q.val:
                search(root.right)
                
            else:
                return

        search(root)
        return LCA[0]
            
#---------------Solution 2 -------------------------------------------------------------------#

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':

        if root.val > p.val and root.val > q.val:
            return self.lowestCommonAncestor(root.left, p, q)
        elif root.val < p.val and root.val < q.val:
            return self.lowestCommonAncestor(root.right, p, q)
        else:
            return root

        



