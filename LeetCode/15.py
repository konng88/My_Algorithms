import numpy as np

def MythreeSum(nums):
    n = len(nums)
    result = []
    nums_ = sorted(nums)
    # print(nums_)

    for i in range(n):
        for j in range(i+1,n-1):
            # print('nums_[',i,']: ',nums_[i],'   nums_[',j,']: ',nums_[j])
            start,end = j+1,n-1
            sum2 = nums_[i] + nums_[j]
            if nums_[start] + sum2 == 0:
                result.append(sorted([nums_[i],nums_[j],nums_[start]]))
                # print('add ',sorted([nums_[i],nums_[j],nums_[start]]))
            elif nums_[end] + sum2 == 0:
                result.append(sorted([nums_[i],nums_[j],nums_[end]]))
                # print('add ',sorted([nums_[i],nums_[j],nums_[end]]))

            else:
                while end - start > 1:
                    mid = (start + end) // 2
                    # print('mid',nums_[mid])
                    # print(sum2)
                    # print(nums_[mid] + sum2)
                    if nums_[mid] + sum2 == 0:
                        result.append(sorted([nums_[i],nums_[j],nums_[mid]]))
                        # print('add ',sorted([nums_[i],nums_[j],nums_[mid]]))
                        break
                    elif nums_[mid] + sum2 > 0:
                        end = mid
                    else:
                        start = mid

                if j - n > -1:
                    if nums[mid] + sum2 == 0 :
                        result.append(sorted([nums[i],nums[j],nums[mid]]))
                        # print('add ',sorted([nums_[i],nums_[j],nums_[mid]]))

    result = sorted(result)
    l = len(result)
    i = 0
    while i < l-1:
        # if result[i][0]==result[i+1][0] and result[i][1] == result[i+1][1] and result[i][2]==result[i+1][2]:
        if result[i] == result[i+1]:
            result.pop(i)
            l -= 1
        else:
            i += 1
    return result

def threeSum(nums):
        result = set()
        dic = {}
        for n in nums:

            dic[n] = dic.get(n, 0) + 1

        nums = sorted([k for k, v in dic.items()])

        for i, n1 in enumerate(nums):
            for n2 in nums[i:]:
                if (n1 == n2) and (dic[n1] < 2):
                    continue

                n3 = 0 - n1 - n2
                if n3 in dic:
                    count = dic[n3]
                    if n1 == n3:
                        count -= 1
                    if n2 == n3:
                        count -= 1

                    if count > 0:
                        result.add(tuple(sorted([n1, n2, n3])))
        return result

print(threeSum([-1, 0, 1, 2, -1, -4]))
