def maxProduct(nums) -> int:
    mx = [0]*len(nums)
    mn = [0]*len(nums)
    mx[0] = nums[0]
    mn[0] = nums[0]
    for i in range(1,len(nums)):
        s = sorted([mx[i-1]*nums[i],mn[i-1]*nums[i]])
        print('s: ',s)
        mx[i] = s[-1]
        mn[i] = s[0]
        print('mx: ',mx)
        print('mn: ',mn)
    return max(mx)


print(maxProduct([2,3,-2,4]))
