"""
Given four lists A, B, C, D of integer values, compute how many tuples (i, j, k, l) there are such that A[i] + B[j] + C[k] + D[l] is zero.

To make problem a bit easier, all A, B, C, D have same length of N where 0 ≤ N ≤ 500. All integers are in the range of -228 to 228 - 1 and the result is guaranteed to be at most 231 - 1.

Example:

Input:
A = [ 1, 2]
B = [-2,-1]
C = [-1, 2]
D = [ 0, 2]

Output:
2

Explanation:
The two tuples are:
1. (0, 0, 0, 1) -> A[0] + B[0] + C[0] + D[1] = 1 + (-2) + (-1) + 2 = 0
2. (1, 1, 0, 0) -> A[1] + B[1] + C[0] + D[0] = 2 + (-1) + (-1) + 0 = 0
"""
def fourSumCount(A,B,C,D):
    output = 0
    CD = {}
    for c in range(len(C)):
        for d in range(len(D)):
            num = C[c] + D[d]
            if num not in CD.keys():
                CD[num] = 1
            else:
                CD[num] += 1
    for a in range(len(A)):
        for b in range(len(B)):
            num = -A[a] - B[b]
            if num in CD.keys():
                output += CD[num]
    # print(output)
    return output

fourSumCount(A = [ 1, 2],B = [-2,-1],C = [-1, 2],D = [ 0, 2])
