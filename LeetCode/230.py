from _binary_Tree import Binary_Tree,TreeNode
"""
Given a binary search tree, write a function kthSmallest to find the kth smallest element in it.

Note:
You may assume k is always valid, 1 ≤ k ≤ BST's total elements.

Example 1:

Input: root = [3,1,4,null,2], k = 1
   3
  / \
 1   4
  \
   2
Output: 1
Example 2:

Input: root = [5,3,6,2,4,null,null,1], k = 3
       5
      / \
     3   6
    / \
   2   4
  /
 1
Output: 3
"""

def kthSmallest(root,k):
    K = [k]
    def inorderTraversal(root):
        if root.left != None:
            inorderTraversal(root.left)
        K[0] -= 1
        if K[0] == 0:
            K.append(root.val)
        if root.right != None:
            inorderTraversal(root.right)
    inorderTraversal(root)
    return K[1]

root = TreeNode(5)
root.left = TreeNode(3)
root.right = TreeNode(6)
root.left.left = TreeNode(2)
root.left.right = TreeNode(4)
root.left.left.left = TreeNode(1)
print(root.val)
print(root.left.val,root.right.val)
print(root.left.left.val,root.left.right.val)
print(root.left.left.left.val)
print(kthSmallest(root,3))
