def mySqrt(x) -> int:
    if x < 2:
        return x
    start = 0
    end = x
    while True:
        mid = int((start+end)/2)
        if mid ** 2 > x:
            end = mid
        elif mid ** 2 < x:
            start = mid
        else:
            return mid
        if end - start <= 1:
            return start

for i in range(0,1000):
    print(i,mySqrt(i))
