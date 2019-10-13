# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


def addTwoNumbers(l1, l2):
    # depth_l1 = 0
    # depth_l2 = 0
    # while l1.next != None:
    #     depth_l1 += 1
    # while l2.next != None:
    #     depth_l2 += 1

    ten = 0
    digit = (l1.val + l2.val) % 10
    l3 = ListNode(digit)
    l = l3
    ten = (l1.val + l2.val) // 10
    while l1.next != None:
        l1 = l1.next
        if l2.next != None:
            l2 = l2.next
            l3.next = ListNode((l1.val + l2.val + ten) % 10 )
            l3 = l3.next
            ten = (l1.val + l2.val + ten) // 10
        else:
            l3.next = ListNode((l1.val + ten) % 10)
            l3 = l3.next
            ten = (l1.val + ten) // 10


    if l2.next != None:
        while l2.next != None:
            l2 = l2.next
            l3.next = ListNode((ten + l2.val) % 10)
            l3 = l3.next
            ten = (ten + l2.val) // 10
        if ten != 0:
            l3.next = ListNode(1)
    else:
        if ten != 0:
            l3.next = ListNode(ten)
    return l

def print_node(node):
    num = ''
    num = num + str(node.val)

    while node.next != None:
        node=node.next
        num = num + str(node.val)

    print(num)


# l1 = ListNode(2)
# l1.next = ListNode(4)
# l1.next.next = ListNode(3)
# l2 = ListNode(5)
# l2.next = ListNode(6)
# l2.next.next = ListNode(4)
l1 = ListNode(3)
l1.next=ListNode(7)
l2=ListNode(9)
l2.next=ListNode(2)
print_node(l1)
print_node(l2)
l3 = addTwoNumbers(l1, l2)
print_node(l3)
