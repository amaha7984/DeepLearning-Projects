# 543. Diameter of Binary Tree
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        largest_diameter = 0

        def height(node):
            nonlocal largest_diameter    # we cannot manipulate global variable without diclaring it as nonlocal
            if node is None:
                return 0

            right_height = height(node.right)
            left_height = height(node.left)  

            diameter = left_height + right_height

            largest_diameter = max(largest_diameter, diameter)

            return 1 + max(left_height, right_height)
        
        height(root)
        return largest_diameter