import numpy as np
from _LinkedList import Node,LinkedList


def reverse_LinkedList(ll):
    node1 = ll.start
    node2 = ll.start.next
    node1.next = None

    while node2 != None:
        node3 = node2.next
        node2.next = node1
        node1 = node2
        node2 = node3

    return LinkedList(node1)


l = np.random.randint(8,size=20)
ll1 = LinkedList(l)
ll1.print()
ll2 = reverse_LinkedList(ll1)
ll2.print()
