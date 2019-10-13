def uniquePaths(m: int, n: int) -> int:
    M = []
    for i in range(0,m):
        r = []
        for j in range(0,n):
            r.append(0)
        M.append(r)
    for i in range(0,m):
        M[i][0] = 1
    for i in range(0,n):
        M[0][i] = 1
    for i in range(1,m):
        for j in range(1,n):
            M[i][j] = M[i][j-1] + M[i-1][j]
    return M[m-1][n-1]


n=uniquePaths(3,7)
print(n)
