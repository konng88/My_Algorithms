from _binary_Tree import Binary_Tree,TreeNode

tree = Binary_Tree([1,None,2]).root
tree.left = None
tree.right.left = TreeNode(3)

def inorderTraversal(root):
    if root == None:
        return
    ans = []
    def go(root):
        if root.left != None:
            go(root.left)
        ans.append(root.val)
        if root.right != None:
            go(root.right)
    go(root)
    return ans

print(inorderTraversal(tree))
