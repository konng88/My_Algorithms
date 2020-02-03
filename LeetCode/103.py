"""
Given a binary tree, return the zigzag level order traversal of its nodes' values. (ie, from left to right, then right to left for the next level and alternate between).

For example:
Given binary tree [3,9,20,null,null,15,7],
    3
   / \
  9  20
    /  \
   15   7
return its zigzag level order traversal as:
[
  [3],
  [20,9],
  [15,7]
]
"""
from _binary_Tree import TreeNode,Binary_Tree

def zigzagLevelOrder(root):
    if not root: return []
    res, temp, stack, flag=[], [], [root], 1
    while stack:
        for i in range(len(stack)):
            node=stack.pop(0)
            temp+=[node.val]
            if node.left: stack+=[node.left]
            if node.right: stack+=[node.right]
        res+=[temp[::flag]]
        temp=[]
        flag*=-1
    print(res)
    return res

def zigzagLevelOrder2(root):
    if root == None:
        return []
    maxDepth = [1]
    def tag(node,depth):
        if node != None:
            node.depth = depth
            if depth > maxDepth[0]:
                maxDepth[0] = depth
            tag(node.left,depth+1)
            tag(node.right,depth+1)
    tag(root,1)
    queue = [root]
    output = []
    for _ in range(maxDepth[0]):
        output.append([])
    while queue != []:
        node = queue.pop(0)
        depth = node.depth
        if node.left != None:
            queue.append(node.left)
        if node.right != None:
            queue.append(node.right)
        if depth % 2 != 0:
            output[depth-1].append(node.val)
        else:
            output[depth-1].insert(0,node.val)
    return output


root = Binary_Tree([1,2,3,4,5,6,7]).root
zigzagLevelOrder(root)
