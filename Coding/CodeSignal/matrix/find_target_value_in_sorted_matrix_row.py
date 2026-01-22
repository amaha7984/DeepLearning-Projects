"""
You are given a matrix of integers where every row and column are sorted in ascending order. Your task is to find the row that contains a specific target value. If the target value doesn't exist, return None.

The expected time complexity is 
O(n+m), where n is the number of rows and m is the number of columns.
"""

def find_row_with_target(matrix: list[list[int]], target: int) -> int | None:
    if not matrix:
        return None
    row = 0
    col = len(matrix[0]) - 1
    
    
    while row < len(matrix) and col >= 0:
        if matrix[row][col] == target:
            return row
        elif matrix[row][col] > target:
            col -= 1
            
        elif matrix[row][col] < target:
            row += 1
    return None
        

