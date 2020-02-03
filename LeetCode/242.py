def isAnagram(s,t):
    return sorted(s) == sorted(t)

print(isAnagram(s = "anagram", t = "nagaram"))
