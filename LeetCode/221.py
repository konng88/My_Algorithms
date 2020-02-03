"""
Given a 2D binary matrix filled with 0's and 1's, find the largest square containing only 1's and return its area.

Example:

Input:

1 0 1 0 0
1 0 1 1 1
1 1 1 1 1
1 0 0 1 0

Output: 4
"""

def maximalSquare(matrix):
    n = len(matrix)
    if n == 0:
        return 0
    m = len(matrix[0])
    if m == 0:
        return 0
    M = []
    for i in range(n):
        d = []
        for j in range(m):
            d.append((0,0,0))
        M.append(d)
    if matrix[0][0] == '1':
        M[0][0] = (1,1,1)




    for i in range(1,n):
        M[i][0] = (1,1,M[i-1][0][2]+1) if matrix[i][0] == '1' else (0,0,0)
    for j in range(1,m):
        M[0][j] = (M[0][j-1][0]+1,1,1) if matrix[0][j] == '1' else(0,0,0)

    for i in range(1,n):
        for j in range(1,m):
            if matrix[i][j] == '1':
                M[i][j] = (M[i][j-1][0]+1,min(M[i-1][j-1][1]+1,M[i][j-1][0]+1,M[i-1][j][2]+1),M[i-1][j][2]+1)
            else:
                M[i][j] = (0,0,0)

    maxS = 0
    a,b = 0,0
    for i in range(n):
        for j in range(m):
            M[i][j] = min(M[i][j])**2
            if maxS < M[i][j]:
                a,b = i,j
                maxS = M[i][j]


    return maxS

def maximalSquare2(matrix):
    n = len(matrix)
    if n == 0:
        return 0
    m = len(matrix[0])
    if m == 0:
        return 0
    M = []
    for i in range(n):
        M.append([0]*m)
    M[0][0] = 1 if matrix[0][0] == '1' else 0
    maxS = M[0][0]
    for i in range(1,m):
        M[0][i] = 1 if matrix[0][i] == '1' else 0
        if M[0][i] > maxS:
            maxS = M[j][i]
    for i in range(1,n):
        M[i][0] = 1 if matrix[i][0] == '1' else 0
        if M[i][0] > maxS:
            maxS = M[i][0]

    for i in range(1,n):
        for j in range(1,m):
            M[i][j] = min(M[i-1][j],M[i][j-1],M[i-1][j-1]) + 1 if matrix[i][j] == '1' else 0
            if M[i][j] > maxS:
                maxS = M[i][j]
    return maxS**2






print(maximalSquare2([['1']]))
