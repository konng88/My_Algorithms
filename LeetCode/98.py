from _binary_Tree import Binary_Tree,TreeNode
"""
Given a binary tree, determine if it is a valid binary search tree (BST).

Assume a BST is defined as follows:

The left subtree of a node contains only nodes with keys less than the node's key.
The right subtree of a node contains only nodes with keys greater than the node's key.
Both the left and right subtrees must also be binary search trees.
"""
def isValidBST(root):
    if root == None:
        return True
    flag = []
    def go(root,upper,lower):
        # print('-'*20)
        # print('ROOT.VAL',root.val)
        # print('UPPER',upper)
        # print('LOWER',lower)
        # if root.left != None:
        #     print('LEFT CHILD',root.left.val)
        # if root.right != None:
        #     print('RIGHT CHILD',root.right.val)
        # print('-'*20)

        if root.left != None:
            if root.left.val >= root.val:
                flag.append(1)
            if root.left.val <= lower:
                # print('LOWER NOT TRUE')
                # print('LOWER',lower)
                # print('LEFT CHILD',root.left.val)
                flag.append(1)
            go(root.left,min(upper,root.val),lower)

        if root.right != None:
            if root.right.val <= root.val:
                flag.append(1)
            if root.right.val >= upper:
                # print('UPPER NOT TRUE')
                # print('UPPER',upper)
                # print('RIGHT CHILD',root.right.val)
                flag.append(1)

            go(root.right,upper,max(lower,root.val))

    go(root,99999999,-99999999)

    return flag == []


def isValidBSTInorderTraverse(root):
    if root == None:
        return True
    a = []
    def inorderTraverse(root):
        if root.left != None:
            inorderTraverse(root.left)
        a.append(root.val)
        if root.right != None:
            inorderTraverse(root.right)
    inorderTraverse(root)
    print(a)
    for i in range(0,len(a)-1):
        if a[i] >= a[i+1]:
            return False
    return True


tree = Binary_Tree([3,1,5,0,2,4,6,None,None,None,3]).root
# tree = Binary_Tree([2,1,3]).root

tree.left.left.left = None
tree.left.left.right = None
tree.left.right.left = None


print(isValidBST(tree))
print(isValidBSTInorderTraverse(tree))
