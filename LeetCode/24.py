from _LinkedList import Node,LinkedList

def swapPairs(head):
    node = head
    if node == None:
        return node
    if node.next == None:
        return node
    new_head = node.next

    while True:
        if node.next == None:
            break
        n1 = node.next
        n2 = node.next.next
        n1.next = node
        if n2 == None:
            node.next = None
            break
        if n2.next == None:
            node.next = n2
        else:
            node.next = n2.next
        node = n2

    return new_head

l1 = LinkedList([1,2,3,4,5,6]).start

l2 = swapPairs(l1)
l2.print()
