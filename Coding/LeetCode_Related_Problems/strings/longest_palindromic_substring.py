# 5. Longest Palindromic Substring


class Solution:
    def longestPalindrome(self, s: str) -> str:
        # babad babbadbabad  
        # dabab dababdabbab

        long_len = 0
        long_pal = ""

        for x in range(len(s)):
            # even length
            l, r = x, x + 1
            while l >= 0 and r < len(s) and s[l] == s[r]:
                if long_len < (r - l + 1):
                    long_len = r - l + 1
                    long_pal = s[l:r+1]
                l -= 1
                r += 1
                
            # odd length 
            l, r = x, x
            while l >= 0 and r < len(s) and s[l] == s[r]:
                if long_len < (r - l + 1):
                    long_len = r - l + 1
                    long_pal = s[l:r+1]
                l -= 1
                r += 1     
                
        return long_pal
