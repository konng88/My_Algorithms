def reverseWords(s: str) -> str:
    new_s = list(s.strip())
    for i in range(1,len(new_s)).__reversed__():
        if new_s[i] == ' ':
            while new_s[i-1] == ' ':
                del new_s[i-1]
                i -= 1
    new_s = ''.join(new_s)
    new_s = new_s.split(' ')
    new_s = list(reversed(new_s))
    new_s = ' '.join(new_s)
    return new_s


s=reverseWords('a good   example')
print(s)
