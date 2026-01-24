"""
You are given an array of n strings. Your task is to find the longest common suffix shared among all strings in the array. 
A suffix is a sequence of letters at the end of a word. For instance, in the word "flying," "ing" is a suffix.

If the given array is empty or there is no common suffix among the strings, your function should return an empty string.

For example, given an array of strings: ["barking", "parking", "starking"], the longest common suffix is "arking".
"""
#---------------O(n^2) solution------------------------------------------------------------------------------------------#
def solution(strs):
    if not strs:
        return ""

    # Reverse all strings
    reversed_strs = []
    for s in strs:
        reversed_strs.append(s[::-1])

    result = []
    i = 0

    while i < len(reversed_strs[0]):
        ch = reversed_strs[0][i]

        for s in reversed_strs[1:]:
            if i >= len(s) or s[i] != ch:
                result.reverse()
                return "".join(result)

        result.append(ch)
        i += 1

    result.reverse()
    return "".join(result)