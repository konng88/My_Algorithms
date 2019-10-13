import numpy as np
import linear_regression as lr


def f1():
    a=np.array([[0.29674555, 0.29674555, 0.29674555],
     [0.29674555, 0.29674555, 0.29674555],
     [0.29674555, 0.29674555, 0.29674555]])
    b=np.array([[1,-0.26828088,0.83586354]])
    print(np.dot(b,a))

def f2():
    a = np.array([1,2,3,4,5,6,7,8,9])
    a = a.reshape((3,3))
    b=np.array([[1,-0.26828088,0.83586354]])
    b=b.reshape((3,1))
    print(b)
    print(b[1])


def f3():
    b=np.array([1,3.26828088,0.83586354]).reshape((3,1))
    print(np.argmax(b))
    print(np.exp(b))
    sum = np.sum(b)
    print(sum)
    print(np.exp(b)/sum)

def f4():
    b = np.random.randint(25, size=(5,5))
    print(b)
    print(b[[0,3]])

def f5():
    b=np.array([[1,2,3],[4,5,6],[7,8,9]])

    a=np.array([1,2,3])
    print(np.dot(a,b))
    # s = [b,b+1,b+2]
    # s=np.array(s)
    # print(s)

def f6():
    a=np.array([1,2,3])
    b=np.array([4,5,6])
    c = []
    c.append(a)
    c.append(b)
    d = np.exp(a)
    print(a)

def f7():
    b=np.random.randint(9,size=(3,3))
    maxs = np.max(b,axis=0)
    exp = np.exp(b-maxs)
    sum = np.sum(exp,axis=1)
    res = exp / sum

    print('----------b----------')
    print(b)
    print('----------maxs----------')
    print(maxs)
    print('----------exp----------')
    print(exp)
    print('----------sum----------')
    print(sum)
    print('----------res----------')
    print(res)

def f8():
    b=np.random.randint(9,size=(4,3))
    print('----------b----------')
    print(b)
    a = -np.random.randint(3,size=3)
    print('----------a----------')
    print(a)
    print('----------np.dot(a,b.T)----------')
    print(np.dot(a,b.T))
    print('----------abs(np.dot(a,b.T))----------')
    print(abs(np.dot(a,b.T)))
    print('----------np.sum(abs(np.dot(a,b.T)))----------')
    print(np.sum(abs(np.dot(a,b.T))))


def f9():
    y = [1,-1,-1,1,-1]
    b=np.random.randint(9,size=(5,3))
    a=np.array([1,2,3])
    c=np.dot(a,b.T)
    d=c*y
    e=np.dot(np.array(y),b)
    print('----------b----------')
    print(b)
    print('----------a----------')
    print(a)
    print('----------np.dot(a,b.T)----------')
    print(c)
    print('----------y*np.dot(a,b.T)----------')
    print(d)
    print('----------y*b----------')
    print(e)

def f10():
    a=np.array([1,-2,3,-8,-7,2])
    print('----------a----------')
    print(a.shape)
    b=np.zeros(len(a))
    print('----------b----------')
    print(b.shape)
    c = np.array(a<=0,dtype = int)
    print('----------c----------')
    print(c)
    d = np.random.randint(9, size=(3,len(a)))
    print('----------d----------')
    print(d)
    e = np.multiply(c,d)
    print('----------e----------')
    print(e)
    f = np.sum(e,axis=1)
    print('----------f----------')
    print(f)


def f11():
    n = 10
    for i in range(0,100):
        print(np.random.choice(n))

def f12():
    b=np.random.randint(9,size=(3,5))
    print('----------b----------')
    print(b)
    a=np.array([1,2,3])
    print('----------a----------')
    print(a)
    c = np.dot(a,b)
    print('----------c----------')
    print(c)

def f13():
    a=np.array([1,2,3])
    b=8
    c=np.max(a,b)
    print(np.append(b,a))

f13()
