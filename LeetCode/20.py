def isValid(s):
    s = list(s)
    if len(s) % 2 != 0:
        return False
    stack = []
    for i in s:
        if i == '(':
            stack.append(0)
        if i == '[':
            stack.append(1)
        if i == '{':
            stack.append(2)
        if i == ')':
            if len(stack) == 0:
                return False
            c = stack.pop(-1)
            if c != 0:
                return False
        if i == ']':
            if len(stack) == 0:
                return False
            c = stack.pop(-1)
            if c != 1:
                return False
        if i == '}':
            if len(stack) == 0:
                return False
            c = stack.pop(-1)
            if c != 2:
                return False
    if len(stack) == 0:
        return True
    return False

print(isValid("()"))
