import weakref
import copy
import sys
import numpy
import objgraph

class Node:
    def __init__(self,num):
        self.num = num
        self.p=None
        self.str=numpy.zeros((1000,1000))
        self.c =[]
    def __del__(self):
        print(str(self.num)+' is being destroyed.')
    def get_self(self):
        return self
    def nPrint(self):
        print("Im Node")
if __name__ == '__main__':
    n1 = Node(1)
    n2 = Node(2)
    n3 = Node(3)
    n4=Node(4)
    objgraph.show_most_common_types()


    print('n1 rc = ', sys.getrefcount(n1))
    # n1.c.append(weakref.proxy(n2))
    # n1.c.append(weakref.proxy(n3))
    n1.c.append(n2)
    n1.c.append(n3)
    # n1.c[0]=n2
    # b=n2
    # n1.c.append(b)
    # n2.p = weakref.proxy(n1)
    # n3.p = weakref.proxy(n1)
    # n2.p = n1
    # n3.p =n1
    n1.c=None
    print('n1 rc = ',sys.getrefcount(n1))
    print('n2 rc = ' , sys.getrefcount(n2))
    print('n3 rc = ' , sys.getrefcount(n3))


    # print(a)
    # for i in range(5,1000000):
    #     n1.c.append(Node(i))
    print('before n1=None')
    print('n1 rc = ', sys.getrefcount(n1))

    n1=None

    print('n1 rc = ', sys.getrefcount(n1))

    print('n2 rc = ' , sys.getrefcount(n2))
    print('n3 rc = ' , sys.getrefcount(n3))
    print(n2)
    print(n3)


    # print('n1 rc = ', sys.getrefcount(n1))
    # print('n2 rc = ', sys.getrefcount(n2))
    # print('n3 rc = ', sys.getrefcount(n3))

    for i in range(10000000000):
        pass