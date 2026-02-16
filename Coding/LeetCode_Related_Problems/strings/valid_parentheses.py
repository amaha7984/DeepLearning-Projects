# 20. Valid Parentheses

class Solution:
    def isValid(self, s: str) -> bool:
        
        cdict = {')':'(',
                 '}':'{',
                 ']':'['
                }
        
        stack = []

        for i in s:
            if i not in cdict:
                stack.append(i)
            else:
                if not stack:
                    return False
                op = stack.pop()
                if op != cdict[i]:
                    return False

        return not stack
