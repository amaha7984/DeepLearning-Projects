# 101. Symmetric Tree
# Definition for a binary tree node.
#class TreeNode:
#    def __init__(self, val=0, left=None, right=None):
#        self.val = val
#        self.left = left
#        self.right = right

class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:

        node = root

        def search(node1, node2):

            if not node1 and not node2:
                return True
            
            elif not node1 or not node2:
                return False
            
            elif node1.val != node2.val:
                return False
            
            return (search(node1.left, node2.right) and
                   search(node1.right, node2.left))
        
        return search(node, node)


            