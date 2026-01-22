"""
You're given a matrix where each row is sorted in ascending order. The columns are also sorted in ascending order. 
This creates a special pattern where the values in the matrix increase as you move right or down but decrease as you move left or up.
Your task is to write a Python function that counts the number of integers in the matrix that are smaller than the given target. The function should return this count as an integer.

The expected complexity is 

O(n+m), where n is the number of rows and m is the number of columns in the matrix.
"""

def count_less_than(matrix, target):
    row_size = len(matrix)
    if row_size == 0:
        return 0
    col_size = len(matrix[0]) - 1
    
    count = 0
    i = 0 
    
    while i < row_size and col_size >= 0:
        if matrix[i][col_size] < target:
            count += (col_size + 1)
            i += 1
        else:
            col_size -= 1
            
    return count
            
            
            
                
   

