"""
Given a binary tree, return the level order traversal of its nodes' values. (ie, from left to right, level by level).

For example:
Given binary tree [3,9,20,null,null,15,7],
    3
   / \
  9  20
    /  \
   15   7
return its level order traversal as:
[
  [3],
  [9,20],
  [15,7]
]
"""
from _binary_Tree import Binary_Tree,TreeNode
def levelorder2(root):
    if root == None:
        return []

    queue = [root]
    maxd = [0]
    def go(root,depth):
        if root != None:
            root.depth = depth
        if root.left != None:
            maxd[0] =max(maxd[0],depth+1)
            go(root.left,depth+1)
        if root.right != None:
            maxd[0] =max(maxd[0],depth+1)
            go(root.right,depth+1)


    go(root,0)
    output = []
    for _ in range(maxd[0]+1):
        output.append([])

    while queue != []:
        node = queue.pop(0)
        if node.left != None:
            queue.append(node.left)
        if node.right != None:
            queue.append(node.right)
        output[node.depth].append(node.val)


    return output

def levelorder(root):
    if root == None: return []
    queue = [root]
    output = []
    while queue:
        tmp = []
        for _ in range(len(queue)):
            node = queue.pop(0)
            tmp.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        output.append(tmp)
    return output

a = Binary_Tree([1,2,3,4,5,6,7]).root
print(levelorder(a))
