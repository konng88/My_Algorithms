def majorityElement(nums) -> int:
    counts = {}
    for i in nums:
        if i not in counts.keys():
            counts[i] = 1
        else:
            counts[i] += 1
    output = []
    max_count = 0
    for num,count in counts.items():
        if count > max_count:
            output = [num]
            max_count = count
        elif count == max_count:
            output.append(num)
    return output

def majorityElement2(nums) -> int:
    counts = {}
    for i in nums:
        if i not in counts.keys():
            counts[i] = 1
        else:
            counts[i] += 1
    output = []

    for num,count in counts.items():
        if count > len(nums)/3:
            output.append(num)
    return output

print(majorityElement([1,1,1,3,3,2,2,2]))
print(majorityElement2([1,2,3]))
