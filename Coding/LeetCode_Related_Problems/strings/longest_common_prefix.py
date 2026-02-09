# 14. Longest Common Prefix

        
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        if len(strs) == 1:
            return strs[0]
        
        elif len(strs) == 0:
            return ""
        
        long_str = ""
        
        min_len = len(strs[0])
        for i in range(1, len(strs)):
            if len(strs[i]) < min_len:
                min_len = len(strs[i])
        
    
        j = 0

        while j < min_len:
            ch = strs[0][j]
            for i in range(1, len(strs)):
                if strs[i][j] != ch:
                    return long_str
            long_str += ch
            j += 1

        return long_str

