from Support.Board2Array import Board2Array as B2A
from Support.OneHotEncoding import OneHotEncode as OHE
import chess

class FenLoad:
    def __init__(self):
        self.ohe = OHE()
        self.b2a = B2A()
        self.board = chess.Board()
    def getBatchSizeDataFor41FeaturesRL(self,fenDatas):
        batchsizeInput = []
        batchsizeOutput = []
        batchsizeResult = []
        preMoves =[]


        whiteInput = []
        whiteOutput =[]
        whiteResult= []

        blackInput = []
        blackOutput = []
        blackResult = []

        for line in fenDatas:
            fenInform= line
            fenInform = fenInform[:-1]
            fenInform = fenInform.split(":") # ':'으로 나눈다. [0] 입력 fen, [1] 명령어, [2] 게임 결과
            for i in range(5):
                preMoves.append(fenInform[2+i][:4])
            #파일에서 받은 fen을 보드에 입력
            self.board.set_fen(fenInform[0])

            # 입력 데이터 전처리
            if self.board.turn:
                whiteInput.append(self.b2a.board2arrayFor41Feature(self.board,preMoves))  # ForGet~은 36개 데이터
                whiteOutput.append(self.convertOutput(fenInform[1]))
                whiteResult.append(self.convertResultForRL(self.board.turn, fenInform[7]))  # 가치망 사용할 때 입력 받음
            else:
                blackInput.append(self.b2a.board2arrayFor41Feature(self.board,preMoves))  # ForGet~은 36개 데이터
                blackOutput.append(self.convertOutput(fenInform[1]))
                blackResult.append(self.convertResultForRL(self.board.turn, fenInform[7]))  # 가치망 사용할 때 입력 받음
            preMoves.clear()

        batchsizeInput.append(whiteInput)
        batchsizeInput.append(blackInput)
        batchsizeOutput.append(whiteOutput)
        batchsizeOutput.append(blackOutput)
        batchsizeResult.append(whiteResult)
        batchsizeResult.append(blackResult)

        return batchsizeInput, batchsizeOutput, batchsizeResult

    def getBatchSizeDataFor41Features(self, filename, batchSize):
        #배치 사이즈만큼 입력 값을 반환


        trainingFile = open(filename, 'r')
        file = open('LearnCount.txt','r')

        COUNT = int(file.readline())

        batchsizeInput = []
        batchsizeOutput = []
        # batchsizeResult = []
        preMoves =[]

        k=0
        for i, line in enumerate(trainingFile): #학습한 이후 부터 다시 시작
            if  COUNT<= i and i < COUNT+batchSize :
                if k%100 == 0:
                    print("\r",k,end="",flush=True)
                k+=1
                fenInform= line
                if fenInform == None:
                    print("\n파일 끝 도착: ",i,"\n")
                    return batchsizeInput, batchsizeOutput
                fenInform = fenInform[:-1]
                fenInform = fenInform.split(":") # ':'으로 나눈다. [0] 입력 fen, [1] 명령어, [2] 게임 결과
                for i in range(5):
                    preMoves.append(fenInform[2+i][:4])
                #파일에서 받은 fen을 보드에 입력
                self.board.set_fen(fenInform[0])

                #입력 데이터 전처리
                batchsizeInput.append(self.b2a.board2arrayFor41Feature(self.board,preMoves))
                batchsizeOutput.append(self.convertOutput(fenInform[1]))
                # batchsizeResult.append(self.convertResult(fenInform[2])) #가치망 사용할 때 입력 받음
                preMoves.clear()
            elif i >= COUNT+batchSize:
                break

        trainingFile.close()
        file.close()
        return batchsizeInput, batchsizeOutput#, batchsizeResult

    def getDataForRL(self,fenDatas):

        batchsizeInput = []
        batchsizeOutput = []
        batchsizeResult = []

        whiteInput = []
        whiteOutput =[]
        whiteResult= []

        blackInput = []
        blackOutput = []
        blackResult = []
        for line in fenDatas:

            fenInform = line
            fenInform = fenInform.split(":")  # ':'으로 나눈다. [0] 입력 fen, [1] 명령어, [2] 게임 결과
            # 파일에서 받은 fen을 보드에 입력
            self.board.set_fen(fenInform[0])

            # 입력 데이터 전처리
            if self.board.turn:
                whiteInput.append(self.b2a.board2array(self.board))  # ForGet~은 36개 데이터
                whiteOutput.append(self.convertOutput(fenInform[1]))
                whiteResult.append(self.convertResultForRL(self.board.turn,fenInform[2])) #가치망 사용할 때 입력 받음
            else:
                blackInput.append(self.b2a.board2array(self.board))  # ForGet~은 36개 데이터
                blackOutput.append(self.convertOutput(fenInform[1]))
                blackResult.append(self.convertResultForRL(self.board.turn, fenInform[2]))  # 가치망 사용할 때 입력 받음

        batchsizeInput.append(whiteInput)
        batchsizeInput.append(blackInput)
        batchsizeOutput.append(whiteOutput)
        batchsizeOutput.append(blackOutput)
        batchsizeResult.append(whiteResult)
        batchsizeResult.append(blackResult)
        return batchsizeInput, batchsizeOutput, batchsizeResult

    def getBatchSizeData(self, filename, batchSize):
        #배치 사이즈만큼 입력 값을 반환


        trainingFile = open(filename, 'r')
        file = open('C:/Users/keon9/Desktop/Project/New_DeepPurple/MakingValueNetwork/LearnCount.txt','r')

        COUNT = int(file.readline())

        batchsizeInput = []
        batchsizeOutput = []
        # batchsizeResult = []
        k=0
        for i, line in enumerate(trainingFile): #학습한 이후 부터 다시 시작
            if  COUNT<= i and i < COUNT+batchSize :
                if k%100 == 0:
                    print("\r",k,end="",flush=True)
                k+=1
                fenInform= line
                if fenInform == None:
                    print("\n파일 끝 도착: ",i,"\n")
                    return batchsizeInput, batchsizeOutput
                fenInform = fenInform[:-1]
                fenInform = fenInform.split(":") # ':'으로 나눈다. [0] 입력 fen, [1] 명령어, [2] 게임 결과

                #파일에서 받은 fen을 보드에 입력
                self.board.set_fen(fenInform[0])

                #입력 데이터 전처리
                batchsizeInput.append(self.b2a.board2arrayForGetBestMove(self.board)) #ForGet~은 36개 데이터
                batchsizeOutput.append(self.convertResultForTanh(fenInform[1]))
                # batchsizeResult.append(self.convertResult(fenInform[2])) #가치망 사용할 때 입력 받음
            elif i >= COUNT+batchSize:
                break

        trainingFile.close()
        file.close()
        return batchsizeInput, batchsizeOutput#, batchsizeResult

    def convertResult(self,result):

        rm = {'1-0': [1, 0, 0, 0], '0-1': [0, 1, 0, 0], '1/2-1/2': [0, 0, 1, 0],'*': [0, 0, 0, 1]}
        # 게임의 끝, ( 백승 = 1, 흑승 = -1, 무승부, 0 )
        convertedResult = rm[result]

        return convertedResult
    def convertResultForTanh(self,result):

        rm = {'1-0': 1, '0-1': -1, '1/2-1/2': 0,'*': 0}
        # 게임의 끝, ( 백승 = 1, 흑승 = -1, 무승부, 0 )
        convertedResult = rm[result]

        return convertedResult
    def convertResultForRL(self,turn, result):
        #흑일때 적용하기 위해 result값을 변경해야한다
        result = self.convertResultForTanh(result)
        # if turn == False:
        #     if result == 1:
        #         return [-1]
        #     elif result == -1:
        #         return [1]
        #     else:
        #         return [result]
        # else:
        #     return [ result]
        return [result]
    def convertOutput(self,output):
        #들어온 output을 One hot으로 변경해서 반환

        uciOneHot = self.ohe.uciMoveToOnehot(output)  # 4096 onehot

        return uciOneHot
