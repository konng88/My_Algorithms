from _binary_Tree import Binary_Tree,TreeNode
"""
Given a binary tree, return the preorder traversal of its nodes' values.

Example:

Input: [1,null,2,3]
   1
    \
     2
    /
   3

Output: [1,2,3]
"""

def preorderTraversal(root):
    output = []
    if root == None:
        return []

    def go(root):
        output.append(root.val)
        if root.left != None:
            go(root.left)
        if root.right != None:
            go(root.right)
    go(root)
    return output

a = TreeNode(1)
a.right = TreeNode(2)
a.right.left = TreeNode(3)
print((preorderTraversal(a)))
