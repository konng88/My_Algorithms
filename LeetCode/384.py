import random
class solution:
    def __init__(self, nums):
        self.data = nums

    def reset(self):

        return self.data


    def shuffle(self):
        l = self.data.copy()
        random.shuffle(l)
        return l

s = solution([1,2,3,4,5,6,7,8,9])
print(s.shuffle())
