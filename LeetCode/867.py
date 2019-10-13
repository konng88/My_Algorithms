def transpose(A):
    B = []
    num_rows = len(A)
    num_columns = len(A[0])
    for j in range(0,num_columns):
        row = []
        for i in range(0,num_rows):
            row.append(A[i][j])
        B.append(row)
    return B


A = [[1,2,3],[4,5,6]]
B = transpose(A)
print(B)
