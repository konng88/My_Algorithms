"""
Given a singly linked list, group all odd nodes together followed by the even nodes. Please note here we are talking about the node number and not the value in the nodes.

You should try to do it in place. The program should run in O(1) space complexity and O(nodes) time complexity.

Example 1:

Input: 1->2->3->4->5->NULL
Output: 1->3->5->2->4->NULL
Example 2:

Input: 2->1->3->5->6->4->7->NULL
Output: 2->3->6->7->1->5->4->NULL
Note:

The relative order inside both the even and odd groups should remain as it was in the input.
The first node is considered odd, the second node even and so on ...
"""
from _LinkedList import Node,LinkedList
# def oddEvenList(head):
#     if head == None or head.next == None or head.next.next == None:
#         return head
#     node = head.next
#     node2 = head.next.next
#     node3 = node2.next
#     head.next = node2
#     node2.next = node
#     node.next = node3
#     if node3 == None:
#         return head
#
#     node = node3
#     id = 5
#     even = head.next
#     while node.next != None:
#         if id % 2 != 0:
#             evenN = even.next
#             nodeN = node.next
#             evenNN = evenN.next
#             nodeNN = nodeN.next
#             even.next = nodeN
#             nodeN.next = evenNN
#             node.next = evenN
#             evenN.next = nodeNN
#
#             even = even.next
#         node = node.next
#         id += 1
#     return head

def oddEvenList(head):
    if head == None or head.next == None or head.next.next == None:
        return head
    n1 = head
    n2 = head.next
    while n2 != None and n2.next != None:
        node = n2.next
        n2.next = n2.next.next
        n3 = n1.next
        n1.next = node
        node.next = n3
        n1 = n1.next
        n2 = n2.next

    return head






a = LinkedList([1,2,3,4,5,6,7,8,9]).start
oddEvenList(a)
print(a)
h = a
while h != None:
    print(h.val)
    h = h.next
