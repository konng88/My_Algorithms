class solution:
    def __init__(self, nums):
        self.data = nums

    def reset(self):

        return self.data


    def shuffle(self):
        rand = random.sample(self.data, k=len(self.data))
        print(list(rand))
        return rand
s = solution([1,2,3,4,5,6,7,8,9])
print(s.shuffle())
