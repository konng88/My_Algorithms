"""
Given a n x n matrix where each of the rows and columns are sorted in ascending order, find the kth smallest element in the matrix.

Note that it is the kth smallest element in the sorted order, not the kth distinct element.

Example:

matrix = [
   [ 1,  5,  9],
   [10, 11, 13],
   [12, 13, 15]
],
k = 8,

return 13.
Note:
You may assume k is always valid, 1 ≤ k ≤ n2.
"""
def kthSmallest(matrix, k):
    stack = [(0,0)]
    i = 0
    dim = (len(matrix),len(matrix[0]))
    waste = set()
    while stack != [] and i < k:
        stack = sorted(stack,key=lambda x:matrix[x[0]][x[1]],reverse=True)
        # print('stack',stack)
        loc = stack.pop(-1)
        # print('waste',waste)
        # print(matrix[loc[0]][loc[1]])
        if loc[0] + 1 < dim[0] and (loc[0]+1,loc[1]) not in waste:
                stack.append((loc[0]+1,loc[1]))
                waste.add((loc[0]+1,loc[1]))
        if loc[1] + 1 < dim[1] and (loc[0],loc[1]+1) not in waste:
                stack.append((loc[0],loc[1]+1))
                waste.add((loc[0],loc[1]+1))

        i += 1

    # stack = sorted(stack,key=lambda x:matrix[x[0]][x[1]])
    # loc = stack[0]
    # print(loc,matrix[loc[0]][loc[1]])
    return matrix[loc[0]][loc[1]]


# a = kthSmallest(matrix = [
#    [ 1,  5,  9],
#    [10, 11, 13],
#    [12, 13, 15]
# ],
# k = 8)

a = kthSmallest(matrix=[[1,2,3,7],[5,10,14,16],[8,10,18,19],[9,12,22,24]],k=14)

print(a)
