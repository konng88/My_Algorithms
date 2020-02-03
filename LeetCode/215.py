"""
Find the kth largest element in an unsorted array. Note that it is the kth largest element in the sorted order, not the kth distinct element.

Example 1:

Input: [3,2,1,5,6,4] and k = 2
Output: 5
Example 2:

Input: [3,2,3,1,2,4,5,5,6] and k = 4
Output: 4
Note:
You may assume k is always valid, 1 ≤ k ≤ array's length.
"""


def findKthLargest(nums, k):
    n = len(nums)
    stack = sorted(nums[0:k],reverse=True)

    for i in range(k,n):
        if nums[i] > stack[-1]:
            stack.pop(-1)
            stack.append(nums[i])
            stack.sort(reverse=True)
    return stack[-1]


findKthLargest([3,2,3,1,2,4,5,5,6] ,k = 4)
