"""
iven a list of daily temperatures T, return a list such that, for each day in the input, tells you how many days you would have to wait until a warmer temperature. If there is no future day for which this is possible, put 0 instead.

For example, given the list of temperatures T = [73, 74, 75, 71, 69, 72, 76, 73], your output should be [1, 1, 4, 2, 1, 1, 0, 0].

Note: The length of temperatures will be in the range [1, 30000]. Each temperature will be an integer in the range [30, 100].
"""
def dailyTemperatures(T):
    n = len(T)
    if n == 0:
        return []
    ts = sorted(enumerate(T),key=lambda x:x[0])
    print(ts)
    stack = [ts[-1]]
    output = [0] * n
    for i in reversed(range(n-1)):
        while stack != [] and stack[-1][1] <= ts[i][1]:
            stack.pop(-1)
        stack.append(ts[i])
        print('stack',stack)
        if len(stack) == 1:
            output[i] = 0
        else:
            output[i] = stack[-2][0] - stack[-1][0]

    print(output)







dailyTemperatures([73, 74, 75, 71, 69, 72, 76, 73])
