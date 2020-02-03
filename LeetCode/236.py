from _binary_Tree import Binary_Tree,TreeNode
"""
Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.

According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between two nodes p and q as the lowest node in T that has both p and q as descendants (where we allow a node to be a descendant of itself).”

Given the following binary tree:  root = [3,5,1,6,2,0,8,null,null,7,4]

Example 1:

Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
Output: 3
Explanation: The LCA of nodes 5 and 1 is 3.
Example 2:

Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
Output: 5
Explanation: The LCA of nodes 5 and 4 is 5, since a node can be a descendant of itself according to the LCA definition.
"""
def lowestCommonAncestor(root, p, q):
    def addCheck(root):
        if root.left != None:
            addCheck(root.left)
        if root.right != None:
            addCheck(root.right)
        root.check = [False,False]


    def go(root):
        if root.left != None:
            go(root.left)
        if root.right != None:
            go(root.right)
        if root.left == None:
            leftCheck = [False,False]
        else:
            leftCheck = root.left.check
        if root.right == None:
            rightCheck = [False,False]
        else:
            rightCheck = root.right.check
        root.check = [leftCheck[0] or rightCheck[0],leftCheck[1] or rightCheck[1]]
        if root == p:
            root.check[0] = True
        if root == q:
            root.check[1] = True
        print(root.val,root.check,leftCheck,rightCheck)





    def find(root):
        if root.check == [True , True]:
            output[0] = root
        if root.left != None:
            find(root.left)
        if root.right != None:
            find(root.right)



    addCheck(root)
    go(root)
    output = [0]
    find(root)
    return output[0]










root = TreeNode(5)
root.left = TreeNode(3)
root.right = TreeNode(6)
root.left.left = TreeNode(2)
root.left.right = TreeNode(4)
root.left.left.left = TreeNode(1)
root.left.left.right = TreeNode(8)
print(root.val)
print(root.left.val,root.right.val)
print(root.left.left.val,root.left.right.val)
print(root.left.left.left.val,root.left.left.right.val)
print('-'*10)
p = root.left.left.right
q = root.left.right
print('target:',p.val,q.val)
node = lowestCommonAncestor(root, p, q)
print(node.val)
