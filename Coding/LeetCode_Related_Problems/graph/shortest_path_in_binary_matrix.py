"""
1091. Shortest Path in Binary Matrix
"""

from collections import deque
class Solution:
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:

        if grid[0][0] == 1 or grid[-1][-1]:
            return -1
        queue = deque([(0, 0, 1)]) # x, y, path_length
        directions = [(0, 1), (0, -1), (-1, 0), (-1, -1), (1, 0), (1, -1), (-1, 1), (1, 1)]
        grid[0][0] = 1

        while queue:
            x, y, path_length = queue.popleft()
            if (x, y) == (len(grid) -1, len(grid[0]) - 1):
                return path_length
            for i, j in directions:
                new_x = x + i
                new_y = y + j

                if (0<= new_x < len(grid)) and (0<= new_y < len(grid[0])) and grid[new_x][new_y] == 0:
                    grid[new_x][new_y] = 1
                    queue.append((new_x, new_y, path_length + 1))

        return -1
        
        