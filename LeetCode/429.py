"""
Given an n-ary tree, return the level order traversal of its nodes' values.

Nary-Tree input serialization is represented in their level order traversal, each group of children is separated by the null value (See examples).



Example 1:



Input: root = [1,null,3,2,4,null,5,6]
Output: [[1],[3,2,4],[5,6]]
"""

class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children

def levelOrder(root):
    output = [[root.val]]
    def go(root,depth):

        if root.children != None:
            if len(output) <= depth:
                output.append([])
            for i in range(len(root.children)):
                output[depth].append(root.children[i].val)
                go(root.children[i],depth+1)
    go(root,1)
    return output

def levelOrderBFS(root):
    queue = [root] if root else []
    ans = []
    while queue:
        ans.append([node.val for node in queue])
        queue_cp = queue
        queue = []
        for node in queue_cp:
            if node.children != None:
                for i in range(len(node.children)):
                    queue.append(node.children[i])

    return ans


nodes = [Node(i) for i in range(0,11)]

root = nodes[1]
root.children = [nodes[3],nodes[2],nodes[4]]
root.children[0].children = [nodes[5],nodes[6]]
output = levelOrderBFS(root)
print(output)
