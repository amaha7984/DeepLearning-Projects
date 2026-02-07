from collections import deque
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        d = deque()

        empty = 0
        fresh = 1
        rotten = 2

        num_fresh = 0

        m, n = len(grid), len(grid[0])

        for i in range(m):
            for j in range(n):
                if grid[i][j] == rotten:
                    d.append((i, j))
                elif grid[i][j] == fresh:
                    num_fresh += 1
        
        if num_fresh == 0:
            return 0
        min_minute = -1
        
        while d:
            min_minute += 1

            for _ in range(len(d)):
                i, j = d.popleft()

                for l, r in [(i, j + 1), (i + 1, j), (i - 1, j), (i, j - 1)]:
                    if 0 <= l < m and 0 <= r < n and grid[l][r] == fresh:
                        grid[l][r] = rotten
                        num_fresh -= 1
                        d.append((l, r))
        if num_fresh == 0:
            return min_minute
        else:
            return -1






