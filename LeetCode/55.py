import numpy as np
"""
self-implemented-version Dynammic Programming
"""
# def canJump(nums) -> bool:
#     if len(nums) == 1:
#         return True
#     can = [True]
#     for i in range(1,len(nums)):
#         can.append(False)
#     def can_reach(pos):
#         if nums[pos] == 0:
#             return
#         max_step = nums[pos]
#         for i in range(1,min(max_step+1,len(nums)-pos)).__reversed__():
#             if can[pos+i] == False:
#                 can[pos+i] = True
#                 if pos+i == len(nums)-1:
#                     return
#                 can_reach(pos+i)
#             else:
#                 return
#
#
#
#     can_reach(0)
#     print(can)
#     return can[-1]

"""
Memo-vresion of Dynammic Programming
"""

# def canJump(nums) -> bool:
#     length = len(nums)
#     memo = []
#     for i in range(0,length):
#         memo.append('Unknown')
#     memo[-1] = 'Good'
#     for i in range(0,length-2).__reversed__():
#
#         furthest = min(length-1,i+nums[i])
#         for j in range(i+1,furthest+1):
#             if memo[j] == 'Good':
#                 memo[i] = 'Good'
#                 break
#     return memo[0] == 'Good'


"""
Greedy-version
"""
def canJump(nums) -> bool:
    if len(nums) == 1:
        return True
    if nums[0] == 0:
        return False
    energy = nums[0]

    for i in range(1,len(nums)-1):
        new_energy = max(energy-1,nums[i])
        if new_energy == 0:
            return False
        energy = new_energy
    return True



a=list(reversed(range(1,25001)))
a.append(1)
a.append(0)
a.append(0)


print(canJump(a))
print(canJump([3,0,0,0]))
