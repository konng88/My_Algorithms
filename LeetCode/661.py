def triangleNumber(nums):
    if len(nums) < 3:
        return 0
    nums.sort()
    count = 0
    for i in range(len(nums)-1,1,-1):
        target = nums[i]
        start = 0
        end = i-1
        while start<end:
            if nums[start] + nums[end] <= target:
                start += 1
            else:
                count += end-start
                end -= 1
    return count

print(triangleNumber([2,2,3,4]))
