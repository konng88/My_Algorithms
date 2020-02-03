def groupAnagrams(strs):
    dic = {}

    for str in strs:
        chars = sorted(list(str))
        chars_ = ''.join(c for c in chars)
        if chars_ not in dic.keys():
            dic[chars_] = [str]
        else:
            dic[chars_].append(str)

    output = list(dic.values())
    return output

# print(groupAnagrams(["eat", "tea", "tan", "ate", "nat", "bat"]))
print(groupAnagrams(["tea","","eat","","tea",""]))
