# Problem 246: strobogrammatic number 

"""
Given a string num which represents an integer, return true if num is a strobogrammatic number.

A strobogrammatic number is a number that looks the same when rotated 180 degrees (looked at upside down).
Example 1:

Input: num = "69"
Output: true
Example 2:

Input: num = "88"
Output: true
Example 3:

Input: num = "962"
Output: false
 

Constraints:

1 <= num.length <= 50
num consists of only digits.
num does not contain any leading zeros except for zero itself.
"""

class Solution:
    def isStrobogrammatic(self, num):

        strobo_numbers = {
            '0': '0',
            '1': '1',
            '8': '8',
            '6': '9',
            '9': '6'
        }
    
    if len(num) == 1:
        return num in ['0', '1', '8']
    
    else:
        i = 0
        j = len(num) - 1

        while i <= j:
            if num[i] in strobo_numbers and strobo_numbers[num[i]] == num[j]:
                i += 1
                j -= 1
            else:
                return False
        return True

        

