from _LinkedList import Node,LinkedList


def mergeTwoLists(l1,l2):
    if l1 == None:
        return l2
    if l2 == None:
        return l1
    n1 = l1
    n2 = l2
    if n1.val > n2.val:
        start = n2
        n2 = n2.next
    else:
        start = n1
        n1 = n1.next
    n = start
    while n1 != None:
        print('n:',n.val,'n1:',n1.val,'n2:',n2.val)
        if n2 != None:
            if n1.val > n2.val:
                n.next = n2
                n2 = n2.next

            else:
                n.next = n1
                n1 = n1.next
            n = n.next
        else:
            n.next = n1
            n1 = n1.next
            return start

    n.next = n2
    return start

L1 = LinkedList([2]).start
L2 = LinkedList([1]).start
s = mergeTwoLists(L1,L2)
while s != None:
    print(s.val)
    s = s.next
