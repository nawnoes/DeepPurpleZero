import MonteCarlo.Tree as TR
# from MonteCarlo.NeuralNetwork import ValueNetwork as VN
# from MonteCarlo.NeuralNetwork import Rollout as RO
import chess
import threading

class MontecarloTreeSearch():
    def __init__(self, searchRepeatNum=500, searchDepth = 20, expandPoint=1000):
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
            updateNode, rolloutResult , valueNetworkResult = self.evaluation()
            self.backpropagation(updateNode,rolloutResult , valueNetworkResult)
        else:
            #여기서 누수가 발생할 수 도 큐에서 들어가는 것이 있고
            #빠져나오지느 못하고 있는 큐가 있다.
            #트리생성 중 게임이 종료되면 실제 결과를 적용
            realResult = self.tree.translatedResult()
            del job
            # print("realResult: ",realResult)
            #시뮬레이션으로 얻은 결과를 보다 크게 점수를 부여 10배
            self.backpropagation(self.tree.get_CurrentNode(),realResult* 10, 0)


    def selection(self, depth):
        if depth > self.searchDepth:
           return True
        if depth == self.searchDepth and self.tree.currentNode.get_visit() >= self.expandPoint:
            #마지막 노드에서 확장 조건을 만족하면
            self.expansion()
            return True
        else:
            self.tree.makeNextChildByPolicyNetwork()

        if depth == self.searchDepth:
            return True
        else:
            return False

    def evaluation(self):
        # print("평가")
        #evaluationQueue에서 하나씩 평가 진행
        job = self.evaluationQueue.pop(0) # job[0]: 평가되어야할 노드, job[1]: 체스 보드
        updateNode = job[0]
        if updateNode.get_N_rollout() == 0 and updateNode.get_N_value() == 0:
            # print("평가 새로 계산")
            valueNetworkResult = self.valueNetwork.get_ValueNetwork(job[1])
            try:
                rolloutResult = self.rolloutSimulation(job[1])
            except:
                rolloutResult = 0
        else:
            # print("평가 재사용")
            # 평가를 재사용하는 경우 값을 update 노드로 부터 받지 않아도 되지 않나 싶다
            valueNetworkResult= 0 #updateNode.get_W_value()
            rolloutResult= 0 #updateNode.get_W_rollout()

        return updateNode, rolloutResult, valueNetworkResult

    def expansion(self):
        #강화 학습된 정책망을 가지고
        #한단계 더 확장. 이때 가장 높은 정책망 값 선택
        self.tree.expand_RL_PolicyNetwork()

    def backpropagation(self,updateNode, rolloutResult, valueNetworkResult):

        if updateNode.is_root():
            return 0
        else:
            if updateNode == None:
                print("update Node None")
            parentNode = updateNode.get_Parent()
            updateNode.renewForBackpropagation(rolloutResult, valueNetworkResult)
            return self.backpropagation(parentNode, rolloutResult, valueNetworkResult)


    def rolloutSimulation(self,chessBoard):
        simulationCount = 0
        tmpBoard = chessBoard.copy()
        while not tmpBoard.is_game_over():
            # print(simulationCount,end="")
            move = self.rollout.get_RolloutMove(tmpBoard)
            tmpBoard.push(chess.Move.from_uci(move))
            simulationCount +=1
            if simulationCount>14:
                # print("롤아웃 결과: ", 0, " 시뮬레이션 수: ", simulationCount)
                return 0
        gameOutput = tmpBoard.result()
        #결과는 1-0, 1/2-1/2, 0-1로 나오므로 변환
        gameOutput = self.convertResult(gameOutput)
        # print("롤아웃 결과: ", gameOutput," 시뮬레이션 수: ",simulationCount)
        return gameOutput
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



