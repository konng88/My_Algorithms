class MyCircularDeque:
    class node:
        def __init__(self):
            self.val = None
            self.next = None
            self.before = None
    def __init__(self, k: int):
        """
        Initialize your data structure here. Set the size of the deque to be k.
        """
        self.start = self.node()
        n0 = self.start
        for i in range(k-1):
            n = self.node()
            n.before = n0
            n0.next = n
            n0 = n
        n.next = self.start
        self.start.before = n
        self.end = self.start
        self.full = k
        self.cap = k
        self.size = 0

    def insertFront(self, value: int) -> bool:
        """
        Adds an item at the front of Deque. Return true if the operation is successful.
        """
        if self.cap > 0:
            if self.size > 0:
                p = self.start.before
                p.val = value
                self.start = p
            elif self.size == 0:
                self.start.val = value
            self.size += 1
            self.cap -= 1
            return True
        return False


    def insertLast(self, value: int) -> bool:
        """
        Adds an item at the rear of Deque. Return true if the operation is successful.
        """
        if self.cap > 0:
            if self.size > 0:
                p = self.end.next
                p.val = value
                self.end = p
            elif self.size == 0:
                self.end.val = value
            self.size += 1
            self.cap -= 1
            return True
        return False


    def deleteFront(self) -> bool:
        """
        Deletes an item from the front of Deque. Return true if the operation is successful.
        """
        if self.cap != self.full:
            if self.size == 1:
                self.start.val = None
            else:
                p = self.start
                p.val = None
                self.start = p.next
            self.size -= 1
            self.cap += 1
            return True
        return False



    def deleteLast(self) -> bool:
        """
        Deletes an item from the rear of Deque. Return true if the operation is successful.
        """
        if self.cap != self.full:
            if self.size == 1:
                self.end.val = None
            else:
                p = self.end
                p.val = None
                self.end = p.before
            self.size -= 1
            self.cap += 1
            return True
        return False


    def getFront(self) -> int:
        """
        Get the front item from the deque.
        """
        if self.size == 0:
            return -1
        return self.start.val

    def getRear(self) -> int:
        """
        Get the last item from the deque.
        """
        if self.size == 0:
            return -1
        return self.end.val

    def isEmpty(self) -> bool:
        """
        Checks whether the circular deque is empty or not.
        """
        return self.size == 0


    def isFull(self) -> bool:
        """
        Checks whether the circular deque is full or not.
        """
        return self.cap == 0

    def getInfo(self):
        print('full:',self.full)
        print('cap:',self.cap)
        print('size:',self.size)
        print('start val:',self.start.val)
        print('end val:',self.end.val)

        p = self.start
        s = ''
        for i in range(self.full):
            s += '-' + str(p.val)
            p = p.next
        print('LinkedList:',s)

        # p = self.end
        # sr = ''
        # for i in range(self.full):
        #     sr += '-' + str(p.val)
        #     p = p.before
        # print('LinkedList Reversed:',sr)

def mytest(k):

    q = MyCircularDeque(k)
    for i in range(k):
        q.insertFront(i+1)
        q.getInfo()
        print()

    print('del front',q.deleteFront())
    q.getInfo()
    print()

    print('del last',q.deleteLast())
    q.getInfo()
    print()

    print('del last',q.deleteLast())
    q.getInfo()
    print()

    print('ins last 7',q.insertLast(7))

    q.getInfo()


def test2():
    circularDeque = MyCircularDeque(3)
    print('circularDeque.insertLast(1)',circularDeque.insertLast(1))
    circularDeque.getInfo()
    print('circularDeque.insertLast(2)',circularDeque.insertLast(2))
    circularDeque.getInfo()

    print('circularDeque.insertFront(3)',circularDeque.insertFront(3))
    circularDeque.getInfo()
    print('circularDeque.insertFront(4)',circularDeque.insertFront(4))
    circularDeque.getInfo()
    print(circularDeque.getRear())
    print(circularDeque.isFull())
    print(circularDeque.deleteLast())
    circularDeque.getInfo()
    print('circularDeque.insertFront(4)',circularDeque.insertFront(4))
    circularDeque.getInfo()
    print('circularDeque.getFront()',circularDeque.getFront())
    circularDeque.getInfo()

def test3():
    q = MyCircularDeque(2)
    print('q.insertFront(7)',q.insertFront(7))
    print(q.getInfo())
    print()
    print('q.deleteLast()',q.deleteLast())
    print(q.getInfo())
    print()
    print('q.getFront()',q.getFront())
    print(q.getInfo())
    print()
    print('q.insertLast(5)',q.insertLast(5))
    print(q.getInfo())
    print()
    print('q.insertFront(0)',q.insertFront(0))
    print(q.getInfo())
    print()
    print('q.getRear()',q.getRear())
    print(q.getInfo())
    print()
    print('q.getFront()',q.getFront())
    print(q.getInfo())
    print()
    print('q.getFront()',q.getFront())
    print(q.getInfo())
    print()
    print('q.getRear()',q.getRear())
    print(q.getInfo())
    print()
    print('q.insertLast(0)',q.insertLast(0))
    print(q.getInfo())
    print()

test3()


# Your MyCircularDeque object will be instantiated and called as such:
# obj = MyCircularDeque(k)
# param_1 = obj.insertFront(value)
# param_2 = obj.insertLast(value)
# param_3 = obj.deleteFront()
# param_4 = obj.deleteLast()
# param_5 = obj.getFront()
# param_6 = obj.getRear()
# param_7 = obj.isEmpty()
# param_8 = obj.isFull()
