def topKFrequent(nums,k):
    dic = {}
    output = []
    for i in nums:
        if i not in dic:
            dic[i] = 1
        else:
            dic[i] += 1
    dic = sorted(dic.items(),key=lambda x:x[1],reverse=True)

    for i in range(0,k):
        output.append(dic[i][0])
    return output


print(topKFrequent(nums = [1,1,1,2,2,3], k = 2))
