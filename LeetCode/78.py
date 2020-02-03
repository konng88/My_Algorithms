"""
Given a set of distinct integers, nums, return all possible subsets (the power set).

Note: The solution set must not contain duplicate subsets.

Example:

Input: nums = [1,2,3]
Output:
[
  [3],
  [1],
  [2],
  [1,2,3],
  [1,3],
  [2,3],
  [1,2],
  []
]
"""
def subsets(nums):
    if nums == []:
        return []
    def go(list,waitlist):
        if waitlist == []:
            return list
        num = waitlist.pop(-1)
        newl = []
        for i in range(len(list)):
            l = list[i].copy()
            l.append(num)
            newl.append(l)
        list += newl
        return go(list,waitlist)

    output = go([[]],nums)
    
    return output

subsets([1,2,3])
