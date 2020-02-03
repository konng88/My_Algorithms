"""
Given an array nums of n integers where n > 1,  return an array output such that output[i] is equal to the product of all the elements of nums except nums[i].

Example:

Input:  [1,2,3,4]
Output: [24,12,8,6]
Note: Please solve it without division and in O(n).

Follow up:
Could you solve it with constant space complexity? (The output array does not count as extra space for the purpose of space complexity analysis.)
"""

def productExceptSelf(nums):
    n = len(nums)
    prods = [1] * n
    p = 1
    for i in range(1,n):
        prods[i] = nums[i-1] * p
        p = prods[i]
    
    p = 1
    for i in range(1,n+1):
        prods[n-i] = prods[n-i] * p
        p = p * nums[n-i]

    return prods

print(productExceptSelf([1,2,3,4]))
