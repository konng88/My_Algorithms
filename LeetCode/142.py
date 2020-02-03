from _LinkedList import Node,LinkedList

def detectCycle(head):
    if head == None:
        return
    end = Node('END')
    while head.next != None:
        if head.next.val == 'END':
            return head
        node = head.next
        head.next = end
        head = node
    return

l = LinkedList([0,1,2,3,4,5,6,7,8,9])
n2 = l.get_Node(2)
n9 = l.get_Node(l.length-1)
n9.next = n2
n0 = l.start
print(detectCycle(n0))
