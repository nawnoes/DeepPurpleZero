import copy
import Support.MyLogger as MYLOGGER

class Queue:
    def __init__(self):
        self.buffer = []
    def push(self,obj):
        self.buffer.append(obj)
    def pop(self):
        return self.buffer.pop(0)
    def at(self,i):
        return self.buffer[i]
    def is_empty(self):
        if len(self.buffer) == 0:
            return True
        else:
            return False
    def len(self):
        return len(self.buffer)
    def getBuffer(self):
        return self.buffer

class BFS:
    #BFS로 탐색하면서 검사하기 위해
    def __init__(self):
        self.preQ = Queue()
        self.postQ= Queue()
    def search(self,rootNode):
        childList = rootNode.get_Child()

        for child in childList:
            self.preQ.push(child)
        while self.repeat():
            pass

    def repeat(self):
        flag = self.putPreQChildsAtPostQ()
        if flag == False:
            return False
        self.printLevel()
        self.changePreQFromPostQ()
        return True

    def putPreQChildsAtPostQ(self):
        if self.preQ.len() == 0:
            return False
        for i in range(self.preQ.len()):
            childsOfPreQ = self.preQ.at(i).get_Child()
            for child in childsOfPreQ:
                self.postQ.push(copy.deepcopy(child))
        return True

    def changePreQFromPostQ(self):
        del self.preQ
        self.preQ = copy.deepcopy(self.postQ)
        self.postQ = Queue()
    def printLevel(self):
        preQBuffer = self.preQ.getBuffer()

        str = ""
        for child in preQBuffer:
            if child.parent != None:
                com = child.parent.command
            else:
                com = "None"
            str += format("|P_Command : %s" % com, "25s")
        MYLOGGER.debuglog(str)

        str = ""
        for child in preQBuffer:
            str += format("|visit : %d" %child.visit,"25s")
        MYLOGGER.debuglog(str)

        str = ""
        for child in preQBuffer:
            str += format("|policyScore : %f" %child.policyScore,"25s")
        MYLOGGER.debuglog(str)
        str = ""
        for child in preQBuffer:
            str += format("|valueScore : %f" % child.valueScore, "25s")
        MYLOGGER.debuglog(str)

        str = ""
        for child in preQBuffer:
            str += format("|command: %s" % child.command, "25s")
        MYLOGGER.debuglog(str)

        MYLOGGER.debuglog("\n")