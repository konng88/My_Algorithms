from _binary_Tree import TreeNode,Binary_Tree
def maxDepth(root):
    if root == None:
        return 0

    def depth(node,d,mxd):
        # print('val',node.val)
        # print('d',d)
        # print('mxd',mxd)
        if node.left != None:
            mxd = depth(node.left,d+1,max(d+1,mxd))
        if node.right != None:
            mxd = depth(node.right,d+1,max(d+1,mxd))
        return mxd
    mxd = depth(root,1,1)
    return mxd
def maxDepth_DP(root):
    if root == None:
        return 0
    return 1 + max(maxDepth_DP(root.right),maxDepth_DP(root.left))




t = Binary_Tree([3,9,20,None,None,15,7]).root
t.left.left = None
t.left.right = None
print('Max Depth = ',maxDepth_DP(t))
print('Max Depth = ',maxDepth(t))
