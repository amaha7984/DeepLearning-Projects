# 257. Binary Tree Paths

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def binaryTreePaths(self, root: Optional[TreeNode]) -> List[str]:

        if not root:
            return []
        
        out = []
        stk = [(root, str(root.val))]

        while stk:
            node, path = stk.pop()

            if not node.left and not node.right:
                out.append(path)
            
            if node.right:
                stk.append((node.right, path + "->" + str(node.right.val)))
            
            if node.left:
                stk.append((node.left, path + "->" + str(node.left.val)))
            

        return out
                   
        
# Time Complexity: DFS will run exactly once for each node, so, O(n)
# Space Complexity: We are using stack.  O(n) if th tree is skewed or O(log n) if the tree is height balanced



            