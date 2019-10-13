def subarraysWithKDistinct(A,K):
    length = len(A)
    if length < K:
        return []
    output = []
    left = 0
    right = 0
    unique = []
    bias = 0

    while len(unique) < K:
        if right == len(A):
            return []
        if A[right] not in unique:
            unique.append(A[right])
        else:
            while unique[0] != A[right]:
                unique.pop(0)
                left += 1
            unique.pop(0)
            left += 1
        right += 1
        print(unique)
        output.append(A[left:right+1].copy())


    print('filled')
    for i in range(right,len(A)):
        if A[i] not in unique:
            left = left + bias + 1
            bias = 0
            unique.pop(0)
            unique.append(A[i])
            if len(unique) == K:
                output.append(A[left:i+1])
        else:
            while unique[0] != A[i]:
                bias += 1
                unique.pop(0)
            bias += 1
            unique.pop(0)
            unique.append(A[i])
            if len(unique) == K:
                output.append(A[left:i+1])

        print(unique,left,A[left:i+1],i)
    print(output)






subarraysWithKDistinct(A=[1,2,1,2,3], K=2)
