#876. Middle of the Linked List

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        length = 0
        curr = head

        while curr:
            length += 1
            curr = curr.next
        
        look = (length // 2) + 1
        
        curr_new = head
        while curr_new:
            look -= 1
            if look == 0:
                return curr_new
                break
            curr_new = curr_new.next
