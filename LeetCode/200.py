"""
Given a 2d grid map of '1's (land) and '0's (water), count the number of islands. An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.

Example 1:

Input:
11110
11010
11000
00000

Output: 1
Example 2:

Input:
11000
11000
00100
00011

Output: 3
"""
import numpy as np
def numIslands(grid):
    m = len(grid)
    if m == 0:
        return 0
    n = len(grid[0])
    if n == 0:
        return 0
    C = []
    for i in range(m):
        C.append([True] * n)
    num = 0
    def floodFill(i,j,C):
        C[i][j] = False
        if i-1 >= 0 and C[i-1][j] and grid[i-1][j]=='1':
            floodFill(i-1,j,C)
        if j-1 >= 0 and C[i][j-1] and grid[i][j-1]=='1':
            floodFill(i,j-1,C)
        if i+1 < m and C[i+1][j] and grid[i+1][j]=='1':
            floodFill(i+1,j,C)
        if j+1 < n and C[i][j+1] and grid[i][j+1]=='1':
            floodFill(i,j+1,C)
    for i in range(m):
        for j in range(n):
            if grid[i][j] == '1' and C[i][j]:
                floodFill(i,j,C)
                num += 1
    return num




print(numIslands([['1','1','0','0','0'],['1','1','0','0','0'],['0','0','1','0','0'],['0','0','0','1','1']]))
print(numIslands([['1','1','1','1','0'],['1','1','0','1','0'],['1','1','0','0','0'],['0','0','0','0','0']]))
