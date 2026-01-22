"""
Given a square matrix grid of integers, your task is to find the minimum and maximum values at the secondary diagonal. The secondary diagonal starts at the top right corner and ends at the bottom left corner.

Return a list of two elements where the first element is the minimum value, and the second is the maximum value that you have found in the diagonal. If the square matrix is empty, return [None, None].

The time complexity of the solution should not exceed 
O(n), where n is the length of the row (or column) in the grid.
"""
def solution(grid):
    if not grid:
        return [None, None]
    
    i = 0
    j = len(grid[0]) - 1
    
    min_val = float("inf")
    max_val = float("-inf")
    
    while i < len(grid) and j >= 0:
        if min_val > grid[i][j]:
            min_val = grid[i][j]
        if max_val < grid[i][j]:
            max_val = grid[i][j]
        i += 1
        j -= 1
    return [min_val, max_val]
        
        
            
