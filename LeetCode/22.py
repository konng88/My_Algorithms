def generateParenthesis(n):
    if n > 0:
        m = [[]] * (n)
        m[0] = [['(',')']]
        for i in range(1,n):
            ps = m[i-1]
            new_ps = []
            for p in ps:
                for loc in range(len(p)):
                    p_copy = p.copy()
                    p_copy.insert(loc,')')
                    p_copy.insert(loc,'(')
                    new_ps.append(p_copy)
            m[i] = []
            for new_p in new_ps:
                if new_p not in m[i]:
                    m[i].append(new_p)

    for i in range(len(m[n-1])):
        m[n-1][i] = "".join(m[n-1][i])
    return m[n-1]

def generateParenthesis_BruteForce(n):
    def generate(A=[]):
        if len(A) == 2*n:
            if valid(A):
                ans.append(''.join(A))
                return
        else:
            A.append('(')
            generate(A)
            A.pop()
            A.append(')')
            generate(A)
            A.pop()
            return

    def valid(A):
        bal = 0
        for c in A:
            if c == '(':
                bal += 1
            else:
                bal -= 1
            if bal < 0:
                return False
        if bal == 0:
            return True
        return False
    ans = []
    generate()
    return(ans)


def generateParenthesis_Backtrack(n):
    ans = []
    def backtrack(A='',left=0,right=0):
        if len(A) == 2*n:
            ans.append(A)

        if left < n:
            backtrack(A+'(',left+1,right)
        if right < left:
            backtrack(A+')',left,right+1)
    backtrack()
    return ans

print(generateParenthesis_Backtrack(3))
# print(generateParenthesis(4))
