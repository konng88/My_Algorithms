class Node:
    def __init__(self,val):
        self.val = val
        self.children = []

class Tree:
    def __init__(self,data):
        if len(data) == 0:
            print('No Data Collected')
            return None

        self.root = nodes[0]
        nodes = []
        for val in data:
            nodes.append(Node(val))
        parent = 0
        for i in range(2,len(nodes)):
            if nodes[i].val != None:
                nodes[parent].append(nodes[i])
            else:
                parent 
