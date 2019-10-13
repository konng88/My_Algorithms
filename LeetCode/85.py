import numpy as np

def maximalRectangle(matrix) -> int:
    r = len(matrix)
    c = len(matrix[0])
    A = []
    for i in range(0,r):
        l=[]
        for j in range(0,c):
            l.append(0)
        A.append(l)
    B = []
    for i in range(0,r):
        l=[]
        for j in range(0,c):
            l.append(0)
        B.append(l)


    for i in range(0,c):
        if matrix[0][i] == '0':
            A[0][i] = 0
        else:
            A[0][i] = 1



    for i in range(1,r):
        for j in range(0,c):
            if matrix[i][j] == '0':
                A[i][j] = 0
            else:
                A[i][j] = A[i-1][j] + 1

    M = 0
    for i in range(0,r):
        for j in range(0,c):
            areas = [0]
            for a in range(0,j+1):
                h = min(A[i][a:j+1])
                areas.append(h * (j-a+1))

            B[i][j] = max(areas)
            if B[i][j] > M:
                M = B[i][j]
    return M


matrix = [
  ["1","0","1","0","0"],
  ["1","0","1","1","1"],
  ["1","1","1","1","1"],
  ["1","0","0","1","0"]
]
a = maximalRectangle(matrix)
print(a)
