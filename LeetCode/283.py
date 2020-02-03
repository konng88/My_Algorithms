def moveZeroes(nums) -> None:
    for i in range(len(nums)):
        count = 0
        while nums[i] == 0 and count < len(nums):
            print(i,nums,nums[i])
            nums.pop(i)
            nums.append(0)
            count += 1


    print(nums)

moveZeroes([0,0,1])
