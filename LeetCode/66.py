def plusOne(digits):
    num = 0
    w = len(digits)
    for i in range(0,w):
        num += digits[i] * 10 ** (w-i-1)
    num = num + 1
    nums = list(str(num))
    for i in range(len(nums)):
        nums[i] = int(nums[i])

    return nums
solu = plusOne([1,2,3])
print(solu)
