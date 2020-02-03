"""
Given a positive integer n, find the least number of perfect square numbers (for example, 1, 4, 9, 16, ...) which sum to n.

Example 1:

Input: n = 12
Output: 3
Explanation: 12 = 4 + 4 + 4.
Example 2:

Input: n = 13
Output: 2
Explanation: 13 = 4 + 9.
"""
import math
def numSquares(n):
    M = [1] * (n+1)
    for i in range(1,n+1):
        s = math.sqrt(i)
        if s % 1 != 0:
            mn = M[i-1] + 1
            # print(i,M[i],mn)
            # print('int(s)+1',int(s)+1)
            for j in range(2,int(s)+1):
                # print(i,"***************",i-j**2)
                if M[i-j**2] + 1 < mn:
                    mn = M[i-j**2] + 1
            M[i] = mn

    return M[n]

print(numSquares(15))
