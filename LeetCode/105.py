from _binary_Tree import TreeNode,Binary_Tree

def buildTree(preorder,inorder):

    if inorder != []:
        rootval = preorder.pop(0)
        idx = inorder.index(rootval)
        root = TreeNode(rootval)
        root.left = buildTree(preorder,inorder[:idx])
        root.right = buildTree(preorder,inorder[idx+1:])
        return root




root = buildTree([3,9,20,15,7],[9,3,15,20,7])
print(root.val)
print(root.left.val,root.right.val)
print(root.left.left,root.left.right,root.right.left.val,root.right.right.val)
