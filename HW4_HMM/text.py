import numpy as np

def f1():
    a = [[[]]*3]*2
    for i in range(2):
        for j in range(3):
            a[i][j] = [i,j]
    print(a)

f1()
