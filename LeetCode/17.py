"""
Given a string containing digits from 2-9 inclusive, return all possible letter combinations that the number could represent.

A mapping of digit to letters (just like on the telephone buttons) is given below. Note that 1 does not map to any letters.



Example:

Input: "23"
Output: ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].
"""
def letterCombinations(digits):
    if digits == []:
        return []
    queue = list(list(digits).__reversed__())
    output = ['']
    dic = {'2':['a','b','c'],'3':['d','e','f'],'4':['g','h','i'],'5':['j','k','l'],'6':['m','n','o'],'7':['p','q','r','s'],'8':['t','u','v'],'9':['w','x','y','z'],'0':[' ']}
    while queue:
        digit = queue.pop(-1)
        tmp = []
        for v in dic[digit]:
            print('digit',digit)
            print('v',v)
            tmpv = [c+v for c in output]

            tmp += tmpv
        output = tmp
    return output

print(letterCombinations('23'))
