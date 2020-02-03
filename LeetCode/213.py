"""
You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed. All houses at this place are arranged in a circle. That means the first house is the neighbor of the last one. Meanwhile, adjacent houses have security system connected and it will automatically contact the police if two adjacent houses were broken into on the same night.

Given a list of non-negative integers representing the amount of money of each house, determine the maximum amount of money you can rob tonight without alerting the police.

Example 1:

Input: [2,3,2]
Output: 3
Explanation: You cannot rob house 1 (money = 2) and then rob house 3 (money = 2),
             because they are adjacent houses.
Example 2:

Input: [1,2,3,1]
Output: 4
Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3).
             Total amount you can rob = 1 + 3 = 4.
"""
def rob(nums):
    n = len(nums)
    if n == 0:
        return 0
    if n < 4:
        return max(nums)
    M = [0] * (n-1)
    N = [0] * (n-1)
    M[0] = nums[0]
    M[1] = max(nums[0],nums[1])
    N[0] = nums[1]
    N[1] = max(nums[1],nums[2])
    for i in range(1,n-1):
        M[i] = max(M[i-1],M[i-2]+nums[i])
        N[i] = max(N[i-1],N[i-2]+nums[i+1])
    return max(N[-1],M[-1])


print(rob([1,2,3,1]))
