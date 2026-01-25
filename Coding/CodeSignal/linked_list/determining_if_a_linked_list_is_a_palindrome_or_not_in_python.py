"""
You are given a singly linked list, and your task is to determine whether the linked list is a palindrome or not. 
A linked list is a palindrome if it reads the same forward and backward. Implement a function is_palindrome(head) that takes a head node of a singly linked list 
(which is the first node in the list) and returns True if the linked list is a palindrome and False otherwise. The expected time complexity for your solution is O(n).

For example, a linked list with the following nodes: 1 -> 2 -> 3 -> 2 -> 1 would return True as it is a palindrome; 
however, for 1 -> 2 -> 3 -> 4, the function would return False because it doesn't read the same in both directions.
"""

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def is_palindrome(head):
    new_list = []
    while head:
        new_list.append(head.val)
        head = head.next
    l = 0
    r = len(new_list) - 1
    while l < r:
        if new_list[l] != new_list[r]:
            return False
        l += 1
        r -= 1
    return True
            

