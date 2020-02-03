from _LinkedList import Node,LinkedList

def reverseKGroup(head,k):
    if k == 1:
        return head
    node = head
    n = head
    for i in range(k):
        if n == None:
            return head
        n = n.next

    def reverse_next_K(l1,k):
        node = l1
        head = node
        n1 = node.next
        n2 = n1.next
        node.next = None
        for i in range(k-1):

            n1.next = node
            if n2 == None:
                return head,n1

            node = n1
            n1 = n2
            n2 = n2.next
        head.next = n1
        return head,node
    h_old = None
    while True:

        tail_ = node
        for i in range(k):
            if tail_ == None:
                return new_head
            tail_ = tail_.next

        h,t = reverse_next_K(node,k)
        if h_old != None:
            h_old.next = t
        else:
            new_head = t
        h_old = h

        node = h.next



    return new_head

def reverse_next_K(l1,k):
    node = l1
    head = node
    n1 = node.next
    n2 = n1.next
    node.next = None
    for i in range(k-1):
        print('node:',node.val)
        n1.next = node
        if n2 == None:
            return head,n1

        node = n1
        n1 = n2
        n2 = n2.next
    head.next = n1
    return head,node




a = [k for k in range(1,6)]
l1 = LinkedList(a)
l1.print()
l = l1.start


s = reverseKGroup(l,3)
print('s:')
s.print()
