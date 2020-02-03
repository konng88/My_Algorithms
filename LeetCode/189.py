def rotate(nums, k) -> None:
    """
    Do not return anything, modify nums in-place instead.
    """

    for i in range(len(nums)-k):
        tmp = nums.pop(0)
        nums.append(tmp)


    print(nums)

rotate([1,2,3,4,5,6,7] ,k = 3)
