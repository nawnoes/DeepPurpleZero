import MonteCarlo.Tree as TR
# from MonteCarlo.NeuralNetwork import ValueNetwork as VN
# from MonteCarlo.NeuralNetwork import Rollout as RO
from MonteCarlo.Node import Node
import chess
import threading
import Support.MyLogger as MYLOGGER
import time
from NeuralNetwork.ResNet import DeepPurpleNetwork as DPN
import gc
from Support.Search import  BFS


class MontecarloTreeSearch():
    def __init__(self,path, searchRepeatNum=150, searchDepth = 400, expandPoint=1000):
        self.path = path
        self.tree = None
        self.searchDepth = searchDepth
        self.expandPoint = expandPoint
        self.searchRepeatNum = searchRepeatNum
        self.evaluationQueue = []
        self.LIMIT = 300 #최대 LIMIT 초 동안 트리탐색

    def set_state(self,Board):
        self.tree.reset_board(Board)
    def renew_tree(self):
        del self.tree
        self.tree = TR.Tree(self.dpn)

    def MCTS(self,chessBoard):
        #몬테카를로 트리탐색을 통해 값을 얻기 전에
        #기존에 저장된 트리를 리셋해야함
        #추후에 트리 상속으로 개선
        self.tree= TR.Tree(self.path)
        self.tree.reset_board(chessBoard)
        # print("몬테카를로 Search 시작")
        print("몬테카를로 전 노드 개수: ", Node.numOfNodes)
        startTime = time.time()


        for i in range(self.searchRepeatNum):
            if i % 10 == 0:
                print("\r%d" % i , end="")
            # MYLOGGER.debuglog("---------%d search---------"%i)
            self.search(chessBoard)
            endTime = time.time()
            if (endTime-startTime)>self.LIMIT:
                break
        nextMove = self.getNextMove()
        print("몬테카를로 후 노드 개수: ", Node.numOfNodes)
        gc.collect()
        self.tree = None

        return nextMove

    def search(self,chessBoard):
        depth = 1
        self.tree.go_root(chessBoard)
        gameOver = self.tree.get_GameOver()
        job =[]
        selectionResult = False
        while not( gameOver or selectionResult or depth>self.searchDepth):
            selectionResult = self.selection(depth)
            depth += 1
            gameOver = self.tree.get_GameOver()
        # logstr = "------------------ game result = "+str(self.tree.translatedResult())+" --------------"
        # MYLOGGER.debuglog(logstr)
        #selection이 끝난 후 트리가 가리키는 마지막 노드의 값을 Queue에 추가
        job.append(self.tree.get_CurrentNode())
        job.append(self.tree.get_currentBoard())
        if not gameOver :
            self.evaluationQueue.append(job)
            updateNode = self.tree.get_CurrentNode()
            value = self.evaluation(updateNode)
            gameResult = 0
            self.backpropagation(updateNode,gameResult,value)
        else:
            #여기서 누수가 발생할 수도, 큐에서 들어가는 것이 있고
            #빠져나오지느 못하고 있는 큐가 있다.
            #트리생성 중 게임이 종료되면 실제 결과를 적용
            realResult = self.tree.translatedResult()
            del job
            # print("realResult: ",realResult)
            self.backpropagation(self.tree.get_CurrentNode(),realResult, 0)


    def selection(self, depth):
        if depth >= self.searchDepth:
           return True
        else:
            self.evaluationAndExpansion()
            return False
    def evaluation(self,node):
        return self.tree.getCurrentNodeValue()
    def evaluationAndExpansion(self):
        self.tree.makeNextChild()
        currentNode = self.tree.get_CurrentNode()
        return currentNode
    def backpropagation(self,updateNode, gameResult,valueNetworkResult):

        if updateNode.is_root():
            return 0
        else:
            if updateNode == None:
                print("update Node None")
            parentNode = updateNode.get_Parent()
            updateNode.renewForBackpropagation(gameResult, valueNetworkResult)
            return self.backpropagation(parentNode, gameResult, valueNetworkResult)
    def convertResult(self,result):
        rm = {'1-0': 1, '0-1': -1, '1/2-1/2': 0,'*': 0}
        # 게임의 끝, ( 백승 = 1, 흑승 = -1, 무승부, 0 )
        convertedResult = rm[result]
        return convertedResult

    def set_state(self, Board):
        self.tree.reset_board(Board)
    def getNextMove(self): #방문자 수가 가장 높은 다음 수 반환
        rootNode = self.tree.get_RootNode()
        index = rootNode.get_maxVisitedChildIndex()
        self.tree.root_Node.print_childInfo()
        # BFS로 루트 노드로 부터 트리의 상태 확인하기
        # bfs = BFS()
        # bfs.search(rootNode)
        return rootNode.child[index].command
    def getNetwork(self):
        self.tree.getNetwork()

if __name__ == "__main__":
    fens = [
    "k5q1/p2r1N2/Pn2R3/R2p2P1/8/1Pp2Q2/2P3P1/6K1 b - - 0 59"]

    mcts = MontecarloTreeSearch()

    for fen in fens:
        print("fen: ",fen)
        chessBoard = chess.Board(fen)
        nextMove = mcts.MCTS(chessBoard)
        print("몬테카를로 트리 탐색 결과 : ", nextMove)



