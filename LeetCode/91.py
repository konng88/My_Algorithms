"""
A message containing letters from A-Z is being encoded to numbers using the following mapping:

'A' -> 1
'B' -> 2
...
'Z' -> 26
Given a non-empty string containing only digits, determine the total number of ways to decode it.

Example 1:

Input: "12"
Output: 2
Explanation: It could be decoded as "AB" (1 2) or "L" (12).
Example 2:

Input: "226"
Output: 3
Explanation: It could be decoded as "BZ" (2 26), "VF" (22 6), or "BBF" (2 2 6).
"""

def numDecodings(s):
    s = [int(d) for d in s]
    M = [0] * len(s)
    if s[0] == 0:
        return 0
    M[0] = 1
    if len(M) > 1:
        num = 10 * s[0] + s[1]
        if num == 0:
            return 0
        if s[1] == 0:
            if s[0] > 2:
                return 0
            M[1] = 1
        elif num > 26:
            M[1] = 1
        else:
            M[1] = 2


    for i in range(2,len(s)):
        num2 = 10 * s[i-1] + s[i]
        if num2 == 0:
            return 0
        elif s[i] == 0:
            if s[i-1] > 2:
                return 0
            M[i] = M[i-2]
        elif s[i-1] == 0:
            M[i] = M[i-1]
        elif num2 > 26:
            M[i] = M[i-1]

        else:
            M[i] = M[i-2] + M[i-1]
    return M[-1]


print(numDecodings('12'))
