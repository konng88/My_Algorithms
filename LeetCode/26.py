def removeDuplicates(nums):
    if nums == []:
        return 0
    num = nums[0]
    p = 0
    for i in range(1,len(nums)):
        if nums[i-p] == num:
            nums.pop(i-p)
            p += 1
        else:
            num = nums[i-p]
    return nums

nums = removeDuplicates([1,1,2])
print(nums)
