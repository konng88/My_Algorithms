"""
Given a string s, partition s such that every substring of the partition is a palindrome.

Return all possible palindrome partitioning of s.

Example:

Input: "aab"
Output:
[
  ["aa","b"],
  ["a","a","b"]
]
"""
def partition(s):
    def isP(s):
        if len(s) == 0 or len(s) == 1:
            return True
        if s[0] == s[-1]:
            return isP(s[1:-1])
        return False

    def dfs(current,s):
        # print(current,s)
        if s == '':
            output.append(current)

        for i in range(1,len(s)+1):
            if isP(s[:i]):
                l = current.copy()
                l.append(s[:i])
                dfs(l,s[i:])
    output = []
    dfs([],s)

    return output







print(partition('aaba'))
