import os
from multiprocessing import Process, Manager
import multiprocessing

class Node:
    def __init__(self,num):
        self.childs = []
        self.parent=None
        self.num = num

    def add_child(self,child):
        self.childs.append(child)
    def add_parent(self,parent):
        self.parent = parent
    def print_num(self):
        print(self.num)
class LinkedNode:
    def __int__(self):
        self.headNode

def makeTree(mpList):
    mpList[0].print_num()
    mpList[1].print_num()
    # print(mpList)
if __name__ == '__main__':
    # a = multiprocessing.Queue.
    manager = Manager()
    node1 = Node(1)
    node2 = Node(2)
    mpList = manager.list()

    mpList.append(node1)
    mpList.append(node2)

    prcss = Process(target=makeTree,args=(mpList,))
    prcss.run()
    # prcss.join()

