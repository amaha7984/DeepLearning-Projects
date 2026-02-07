# 13. Roman to Integer

class Solution:
    def romanToInt(self, s: str) -> int:
        hmap = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        summ = 0
        i = 0
        while i < len(s):

            if i < len(s) - 1 and hmap[s[i + 1]] > hmap[s[i]]:
                summ += (hmap[s[i + 1]] - hmap[s[i]])
                i += 2
            else:
                summ += hmap[s[i]]
                i += 1
            
        return summ

        
