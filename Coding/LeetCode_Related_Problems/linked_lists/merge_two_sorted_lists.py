#21. Merge Two Sorted Lists
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:

        if list1 is None:
            return list2
        
        if list2 is None:
            return list1

        curr1 = list1
        curr2 = list2
        dummy = ListNode()
        out = dummy

        while curr1 and curr2:
            if curr1.val < curr2.val:
                out.next = curr1
                out = out.next
                curr1 = curr1.next
            elif curr1.val > curr2.val:
                out.next = curr2
                out = out.next
                curr2 = curr2.next
            else:
                out.next = curr1
                out = out.next
                curr1 = curr1.next
        
        while curr1:
            out.next = curr1
            out = out.next
            curr1 = curr1.next
        
        while curr2:
            out.next = curr2
            out = out.next
            curr2 = curr2.next
        
        return dummy.next
        