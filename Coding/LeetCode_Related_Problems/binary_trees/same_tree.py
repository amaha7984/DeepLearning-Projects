# 100. Same Tree
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:

        node1 = p
        node2 = q

        def search(node1, node2):
            if not node1 and not node2:
                return True

            if node2 and not node1:
                return False
            if node1 and not node2:
                return False
            
            if node1.val != node2.val:
                return False
            
            return (search(node1.left, node2.left) and 
                   search(node1.right, node2.right))
        
        return search(node1, node2)

#-------------Another Solution----------#
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:

        if not p and not q:
            return True
        
        elif not p or not q:
            return False
        
        elif p.val != q.val:
            return False
        
        return (self.isSameTree(p.left, q.left) and 
                self.isSameTree(p.right, q.right))



            