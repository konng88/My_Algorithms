import numpy as np
class TreeNode:
    def __init__(self,x):
        self.val = x
        self.left = None
        self.right = None

class Binary_Tree:
    def __init__(self,data):
        if data == []:
            return None
        depth = int(np.ceil(np.log(len(data))/np.log(2)))
        nodes = []
        for i in data:
            nodes.append(TreeNode(i))
        for i in range(1,len(data)):
            d = int(np.floor(np.log(i+1)/np.log(2)))
            bias = int(i - 2**d + 1)
            bias_of_parent = int(np.floor(bias/2))
            if i % 2 == 1:
                nodes[2**(d-1)-1+bias_of_parent].left = nodes[i]
            else:
                nodes[2**(d-1)-1+bias_of_parent].right = nodes[i]
        self.root = nodes[0]

    def __print__(self):
        def print(self,n):
            print(n.val)
            if n.left != None:
                self.print(n.left)
            if n.right != None:
                self.print(n.right)



# null = None
# BT = Binary_Tree([3,9,20,null,null,15,7])
# print(BT)
# print(BT.root.left.right.left.val)
