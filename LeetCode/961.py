class Solution:
    def repeatedNTimes(self, A) -> int:
        repeated = {}
        for i in A:
            if i not in repeated:
                repeated[i] = 1
            else:
                repeated[i] += 1
        for key,value in repeated.items():
            if value == len(A)/2:
                return key

s=Solution()
a = s.repeatedNTimes([1,2,3,3])
print(a)
a = s.repeatedNTimes([2,1,2,5,3,2])
print(a)
a = s.repeatedNTimes([5,1,5,2,5,3,5,4])
print(a)
