import os
from multiprocessing import Process, Manager,Lock
import multiprocessing
import time

class Node:
    def __init__(self,num):
        self.childs = None
        self.parent=None
        self.num = num

    def set_child(self,child):
        self.childs= child
    def set_parent(self,parent):
        self.parent = parent
    def print_num(self):
        print(self.num)
class LinkedNode:

    def __init__(self):
        self.manager = Manager()
        self.managerList = self.manager.list()
        self.headNode=[]
        self.lock = Lock()

    def makeLink(self,no):
        count = 100
        parent =None
        print(no," 번째 프로세스 시작")
        for i in range(count):

            child = Node(i*no)
            child.print_num()
            self.sleep(1)
            if i ==0:
                with self.lock:
                    self.headNode.append(child)
                    print("헤드노드 ",self.headNode)
            if i ==9:
                print("!!")
            child .set_parent(parent)
            if parent:
                parent.set_child(child)
            parent = child
    def makeLinkWithMP(self):
        noOfProcess = 4
        # while
        prcss =[]
        procs = [Process(target=self.makeLink, args=(i,)) for i in range(0,11)]

        for p in procs: p.start()
        for p in procs: p.join()

        # print("헤드 노드 자식 개수: ", len(headNode))
        # for i in range(0,4):
        #     prcss.append( Process(target=self.makeLink, args=(i,)) )
        # # for i in range(len(prcss)):
        # print(1," process RUN")
        # prcss[1].run()
        # print(2, " process RUN")
        # prcss[2].run()
        # print(3, " process RUN")
        # prcss[3].run()
        # print(4, " process RUN")
        # prcss[4].run()

    def sleep(self,second):
        time.sleep(second)
    def printLink(self):
        node = self.headNode
        while node:
            node.print_num()
            node = node.childs


def makeTree(mpList):
    mpList[0].print_num()
    mpList[1].print_num()
    # print(mpList)
if __name__ == '__main__':
    ln = LinkedNode()
    # ln.makeLink(2)
    ln.makeLinkWithMP()
    # ln.printLink()
    # a = multiprocessing.Queue.
    # manager = Manager()
    # node1 = Node(1)
    # node2 = Node(2)
    # mpList = manager.list()
    #
    # mpList.append(node1)
    # mpList.append(node2)
    #
    # prcss = Process(target=makeTree,args=(mpList,))
    # prcss.run()
    # prcss.join()