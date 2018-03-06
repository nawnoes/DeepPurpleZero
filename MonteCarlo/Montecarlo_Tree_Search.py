import MonteCarlo.Tree as TR
# from MonteCarlo.NeuralNetwork import ValueNetwork as VN
# from MonteCarlo.NeuralNetwork import Rollout as RO
import chess
import threading

class MontecarloTreeSearch():
    def __init__(self, searchRepeatNum=500, searchDepth = 10, expandPoint=1000):
        self.tree = TR.Tree()
        self.searchDepth = searchDepth
        self.expandPoint = expandPoint
        self.searchRepeatNum = searchRepeatNum
        self.evaluationQueue = []

    def set_state(self,Board):
        self.tree.reset_board(Board)

    def MCTS(self,chessBoard):
        #몬테카를로 트리탐색을 통해 값을 얻기 전에
        #기존에 저장된 트리를 리셋해야함
        #추후에 트리 상속으로 개선
        self.tree.reset_board(chessBoard)
        # print("몬테카를로 Search 시작")
        for i in range(self.searchRepeatNum):
            if i % 10 == 0:
                print("\r%d" % i , end="")
            self.search(chessBoard)
        nextMove = self.getNextMove()

        return nextMove

    def search(self,chessBoard):
        depth = 2
        self.tree.go_root(chessBoard)
        gameOver = self.tree.get_GameOver()
        job =[]
        selectionResult = False
        while not( gameOver or selectionResult):
            selectionResult = self.selection(depth)
            depth +=1
            gameOver = self.tree.get_GameOver()
        #selection이 끝난 후 트리가 가리키는 마지막 노드의 값을 Queue에 추가
        job.append(self.tree.get_CurrentNode())
        job.append(self.tree.get_currentBoard())
        if not gameOver:
            self.evaluationQueue.append(job)
            updateNode = self.evaluationAndExpansion()
            gameResult = 0
            self.backpropagation(updateNode,gameResult)
        else:
            #여기서 누수가 발생할 수 도 큐에서 들어가는 것이 있고
            #빠져나오지느 못하고 있는 큐가 있다.
            #트리생성 중 게임이 종료되면 실제 결과를 적용
            realResult = self.tree.translatedResult()
            del job
            # print("realResult: ",realResult)
            #시뮬레이션으로 얻은 결과를 보다 크게 점수를 부여 10배
            self.backpropagation(self.tree.get_CurrentNode(),realResult*10, 0)


    def selection(self, depth):
        if depth > self.searchDepth:
           return True
        else:
            self.evaluationAndExpansion()

        if depth == self.searchDepth:
            return True
        else:
            return False

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
            return self.backpropagation(parentNode, gameResult, updateNode.get_valueScore)
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
        return rootNode.child[index].command

if __name__ == "__main__":
    fens = [
    "k5q1/p2r1N2/Pn2R3/R2p2P1/8/1Pp2Q2/2P3P1/6K1 b - - 0 59"]

    mcts = MontecarloTreeSearch()

    for fen in fens:
        print("fen: ",fen)
        chessBoard = chess.Board(fen)
        nextMove = mcts.MCTS(chessBoard)
        print("몬테카를로 트리 탐색 결과 : ", nextMove)



