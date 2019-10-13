def lengthOfLongestSubstring(s):
    ls = list(s)
    queue = []
    max_unique = 0
    for i in range(0,len(s)):

        if ls[i] not in queue:
            queue.append(ls[i])
            if len(queue) > max_unique:
                max_unique = len(queue)
        else:
            j = 0
            while queue[0] != ls[i]:
                queue.pop(0)
                j += 1
            queue.pop(0)
            queue.append(ls[i])
        # print(queue)

    return max_unique



lengthOfLongestSubstring("pwwkew")
lengthOfLongestSubstring("ohvhjdml")
