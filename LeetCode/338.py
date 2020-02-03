"""
Given a non negative integer number num. For every numbers i in the range 0 ≤ i ≤ num calculate the number of 1's in their binary representation and return them as an array.

Example 1:

Input: 2
Output: [0,1,1]
Example 2:

Input: 5
Output: [0,1,1,2,1,2]
"""
def countBits(num):
    if num == 0:
        return [0]
    output = [0] + [1] * (num)
    move = 1
    i = 2
    while True:
        for _ in range(move):
            if i > num:
                return output
            output[i] = (output[i-move])
            print(output)
            i += 1

        for _ in range(move):
            if i > num:
                return output
            output[i] = (output[i-move] + 1)
            print(output)
            i += 1

        move = move*2


print(countBits(8))
