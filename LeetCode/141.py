from _LinkedList import Node,LinkedList

def hasCycle(head):
    end = Node('END')
    while head != None:
        if head.val == 'END':
            return True
        node = head.next
        head.next = end
        head = node
    return False

l = LinkedList([0,1,2,3,4,5,6,7,8,9])
n2 = l.get_Node(2)
n9 = l.get_Node(l.length-1)
# n9.next = n2
n0 = l.start
print(hasCycle(n0))
