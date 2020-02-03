"""
Given a binary tree, find its minimum depth.

The minimum depth is the number of nodes along the shortest path from the root node down to the nearest leaf node.

Note: A leaf is a node with no children.

Example:

Given binary tree [3,9,20,null,null,15,7],

    3
   / \
  9  20
    /  \
   15   7
return its minimum depth = 2.
"""
from _binary_Tree import Binary_Tree,TreeNode
class Solution:
    def minDepth(self,root):
        if root == None:
            return 0
        if root.left == None or root.right == None:
            return self.minDepth(root.right) + self.minDepth(root.left) + 1
        return 1+min(self.minDepth(root.right),self.minDepth(root.left))


root = Binary_Tree([3,9]).root
root.left.left = None
root.left.right = None
sol = Solution()
print(sol.minDepth(root))
