# Valid Palindrome

class Solution:
    def isPalindrome(self, s: str) -> bool:
        # lenth 0
        # Odd Length 
        # even length
        alphanumeric_list = []
        for i in range(len(s)):
            if s[i].isalnum():
                alphanumeric_list.append(s[i].lower())
        if len(alphanumeric_list) == 0 or len(alphanumeric_list) == 1:
            return True
        i = 0
        j = len(alphanumeric_list) - 1
        while j > i:
            if alphanumeric_list[j] == alphanumeric_list[i]:
                i += 1
                j -= 1
            elif alphanumeric_list[j] != alphanumeric_list[i]:
                return False
        return True

#-----------Same structure code-----------------------------#
class Solution:

    def alpha_numeric(self, c):
        return (ord('A') <= ord(c) <= ord('Z') or
               ord('a') <= ord(c) <= ord('z') or
               ord('0') <= ord(c) <= ord('9'))
        
    def isPalindrome(self, s: str) -> bool:
        # alpha numberic is A - Z, a - z, 0 - 9
        i = 0
        j = len(s) - 1

        while i < j:

            while i < j and not self.alpha_numeric(s[i]):
                i += 1
            while j > i and not self.alpha_numeric(s[j]):
                j -= 1

            if s[i].lower() != s[j].lower():
                return False
            i += 1
            j -= 1
        return True

        