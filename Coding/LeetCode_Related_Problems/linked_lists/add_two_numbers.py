#---------------------------------------2. Add Two Numbers-----------------------------------------------------#
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        a = ""
        b = ""

        head_1 = l1
        while head_1:
            a += str(head_1.val)
            head_1 = head_1.next
        
        head_2 = l2
        while head_2:
            b += str(head_2.val)
            head_2 = head_2.next
        
        a = a[::-1]
        b = b[::-1]

        c = str(int(a) + int(b))
        c = c[::-1]
        l3 = ListNode() #creating dummy node with value = 0
        head = l3
        for i in range(len(c)):
            head.next = ListNode(int(c[i]))
            head = head.next
        return l3.next
