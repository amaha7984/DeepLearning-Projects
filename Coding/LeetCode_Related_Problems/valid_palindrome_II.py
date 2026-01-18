# 680. Valid Palindrome II

class Solution:
    def validPalindrome(self, s: str) -> bool:
        i = 0
        j = len(s) - 1

        while i < j:
            if s[i] != s[j]:
                skipL = s[i+1:j+1]
                skipR = s[i:j]
                if skipL  == skipL[::-1]:
                    return True
                elif skipR == skipR[::-1]:
                    return True
                else:
                    return False
            i += 1
            j -= 1
        return True