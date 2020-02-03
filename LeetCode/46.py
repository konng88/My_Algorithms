"""
Given a collection of distinct integers, return all possible permutations.

Example:

Input: [1,2,3]
Output:
[
  [1,2,3],
  [1,3,2],
  [2,1,3],
  [2,3,1],
  [3,1,2],
  [3,2,1]
]
"""
def permute(nums):
    n = len(nums)
    if n == 0:
        return []
    def go(list,waitlist):
        # print(list,waitlist)
        if waitlist == []:
            return list

        num = waitlist.pop(-1)
        output = []
        for i in range(len(list)):
            for j in range(len(list[i])):
                tmp = list[i].copy()
                tmp.insert(j,num)
                output.append(tmp)
            list[i].append(num)
            output.append(list[i])
        return go(output,waitlist)
    # tmp = nums.pop(-1)
    #
    # go([[tmp]],nums)
    return go([[]],nums)


print(permute([1,2,3]))
