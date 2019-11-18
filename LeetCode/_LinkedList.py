import numpy as np
class Node:
    def __init__(self,x):
        self.val = x
        self.next = None

class LinkedList:
    def __init__(self,l):
        if type(l) == list or type(l) == np.ndarray:
            self.start = Node(l[0])
            self.length = len(l)
            node = self.start
            for i in range(1,self.length):
                node.next = Node(l[i])
                node = node.next
        elif type(l) == Node:
            self.start = l
        else:
            print('unknown data type')

    def print(self):
        output = ''
        node = self.start
        while node != None:
            if node.next == None:
                output += str(node.val)
            else:
                output += (str(node.val) + '-')
            node = node.next
        print(output)
