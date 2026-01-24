"""
You are given two strings, string1 and string2. Your goal is to determine a new string, string3, that is formed by characters that occur in both string1 and string2 in the same order as they occur in string1.
Characters in string3 should maintain their original sequence order from string1. If a character is repeated in string1 and string2, include that character in string3 as many times as it occurs in both strings, 
but not more than that. For example, given string1 = "apple" and string2 = "peach", the resulting string3 would be "ape".
Your algorithm should not exceed a time complexity of O(string1.length+string2.length).
"""
#---------------O(n^2) solution------------------------------------------------------------------------------------------#
def solution(string1, string2):
    string2 = list(string2)
    string3 = []
    
    for i in string1:
        j = 0
        while j < len(string2):
            if i == string2[j]:
                string3.append(string2[j])
                del string2[j]
                break
            j += 1
    return ''.join(string3)

#--------------------------------------O(string1.length+string2.length)----------------------------------------------------#
from collections import Counter
def solution(string1, string2):
    freq = Counter(string2)
    string3 = []
    for i in string1:
        if freq[i] > 0:
            string3.append(i)
            freq[i] -= 1
    return "".join(string3)
            
    