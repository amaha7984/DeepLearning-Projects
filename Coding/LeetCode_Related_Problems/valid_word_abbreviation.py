"""
408. Valid Word Abbreviation
Easy
Company Tags
A string can be shortened by replacing any number of non-adjacent, non-empty substrings with their lengths (without leading zeros).

For example, the string "implementation" can be abbreviated in several ways, such as:

"i12n" -> ("i mplementatio n")
"imp4n5n" -> ("imp leme n tatio n")
"14" -> ("implementation")
"implemetation" -> (no substrings replaced)
Invalid abbreviations include:

"i57n" -> (i mplem entatio n, adjacent substrings are replaced.)
"i012n" -> (has leading zeros)
"i0mplementation" (replaces an empty substring)
You are given a string named word and an abbreviation named abbr, return true if abbr correctly abbreviates word, otherwise return false.

A substring is a contiguous non-empty sequence of characters within a string.
"""
class Solution:
    def validWordAbbreviation(self, word: str, abbr: str) -> bool:
        i, j = 0, 0

        while i < len(word) and j < len(abbr):
            if word[i] == abbr[j]:
                i += 1
                j += 1
            elif abbr[j].isalpha() or abbr[j] =="0":
                return False
            
            else:
                skip = 0
                while j < len(abbr) and not abbr[j].isalpha():
                    skip = skip * 10 + int(abbr[j])    
                    j += 1

                i += skip

        return i == len(word) and j == len(abbr)

        