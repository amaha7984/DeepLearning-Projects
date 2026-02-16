# 1047. Remove All Adjacent Duplicates In String

#-----------------------Two Pointers Solution---------------------------------------#
class Solution:
    def removeDuplicates(self, s: str) -> str:
        if len(s) <= 1:
            return s
        #l, r 
        s = list(s)
        l = 0
        r = 1

        while r < len(s):
            if s[l] != s[r]:
                l += 1
                r += 1
            elif s[l] == s[r]:
                del s[l:r+1]

                if l > 0:
                    l -= 1
                r = l + 1


        return "".join(s)
    
#-----------------------Stack Solution---------------------------------------#
class Solution:
    def removeDuplicates(self, s: str) -> str:
        if len(s) <= 1:
            return s

        stk = []
        stk.append(s[0])

        for i in range(1, len(s)):
            if stk and stk[-1] == s[i]:
                stk.pop()
            else:
                stk.append(s[i])

        return "".join(stk)

            

            

        
