class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        hset = set()
        l = 0
        sub_length = 0
        for r in range(len(s)):
            while s[r] in hset:
                hset.remove(s[l])
                l += 1
            hset.add(s[r])

            length = r - l + 1
            sub_length = max(length, sub_length)
        return sub_length



        