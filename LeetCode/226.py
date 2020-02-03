from _binary_Tree import Binary_Tree,TreeNode
"""
Invert a binary tree.

Example:

Input:

     4
   /   \
  2     7
 / \   / \
1   3 6   9
Output:

     4
   /   \
  7     2
 / \   / \
9   6 3   1
"""
def invertTree(root):
    if root != None:
        node = root.left
        root.left = root.right
        root.right = node
        invertTree(root.left)
        invertTree(root.right)
        return root

root = Binary_Tree([4,2,7,1,3,6,9]).root
print(root.val)
print(root.left.val,root.right.val)
print(root.left.left.val,root.left.right.val,root.right.left.val,root.right.right.val)
invertTree(root)
print(root.val)
print(root.left.val,root.right.val)
print(root.left.left.val,root.left.right.val,root.right.left.val,root.right.right.val)
