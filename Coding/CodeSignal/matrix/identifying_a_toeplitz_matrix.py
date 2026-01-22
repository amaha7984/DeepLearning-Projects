"""
You are given a square matrix of n x n size. Your task is to write a Python function that indicates whether the matrix is a Toeplitz matrix.

In a Toeplitz matrix, each descending diagonal (from left to right) is constant. That is, elements in each descending diagonal are the exact same.
"""

from typing import List

def is_toeplitz(matrix: List[List[int]]) -> bool:
   
   size = len(matrix)
   for i in range(1, size):
      for j in range(1, size):
         if matrix[i][j] != matrix[i - 1][j - 1]:
            return False
   return True
   

