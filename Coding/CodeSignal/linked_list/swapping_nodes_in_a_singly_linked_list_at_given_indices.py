"""
You are given a singly linked list and two indices, start and end (both indices are 0-based). Write a Python function 
swap_linked_list_nodes(head: ListNode, start: int, end: int) -> ListNode that swaps the nodes of the linked list at these two provided indices. 
The function should return the head node of the modified linked list. When swapping, you should only change the next property of a node, not the actual node values. It is guaranteed that start <= end.

For example, consider the linked list 17 -> 2 -> 13 -> 4 -> 51 -> 22 -> 33 -> 84 -> 5 and you are given start = 2 and end = 6. 
"""

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val     # Holds the value or data of the node
        self.next = next  # Points to the next node in the linked list; default is None

def swap_linked_list_nodes(head, start, end):
    if start == end or not head:
        return head

    prev = None
    curr = head
    idx = 0

    prev_start = None
    start_node = None
    prev_end = None
    end_node = None

    # 1) Traverse once and capture needed nodes
    while curr:
        if idx == start:
            prev_start = prev
            start_node = curr
        if idx == end:
            prev_end = prev
            end_node = curr
            break
        prev = curr
        curr = curr.next
        idx += 1

    # 2) Rewire previous nodes
    if prev_start:
        prev_start.next = end_node
    else:
        head = end_node   # start == 0

    if prev_end:
        prev_end.next = start_node

    # 3) Swap next pointers of start and end nodes
    start_node.next, end_node.next = end_node.next, start_node.next

    return head

