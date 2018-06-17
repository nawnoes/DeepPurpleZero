import ReinforcementLearning.ChessAI as AI
from  NeuralNetwork.ResNet import DeepPurpleNetwork as DPN
import chess
import copy
import Support.FenLoad as FL
import os
import shutil
import tensorflow as tf
import objgraph
from Support.Record import GameInfo
# import ChessBoard

class Play:
    def __init__(self):
        self.gameInfo = GameInfo()
        g = tf.Graph()
        postCheckpointPath = '../Checkpoint/Later/'
        preCheckpointPath = '../Checkpoint/Former/'
        trainCheckpointPath = postCheckpointPath
        self.LaterAI = AI.ChessAI(postCheckpointPath)
        self.FormerAI = AI.ChessAI(preCheckpointPath)
        with g.as_default(): #강화학습을 위한 그래프
            self.trainNetwork = DPN(trainCheckpointPath,is_traing=True)

        self.loadFenData= FL.FenLoad()
        self.gameInfo.load()
    def __del__(self):
        print("")

    def saveRLData(self, chessBoard, result):
        # 파일 경로에 주어진 data를 저장
        datas = []

        try:
            f = open('../Data/RLData.txt', 'a')
        except:
            f = open('../Data/RLData.txt', 'w')

        while len(chessBoard.move_stack) != 0:
            command = chessBoard.pop().__str__()
            fen = chessBoard.fen()

            datas.append(fen + ":" + command + ":" + result)

        copyDatas = copy.deepcopy(datas)

        for i in range(len(datas)):
            # 데이터를 역순으로 저장
            data = datas.pop() + "\n"
            f.write(data)

        f.close()

        return copyDatas

    def reinforcementLearning(self, fenDatas, turn):
        input, output, result = self.loadFenData.getDataForRL(fenDatas)
        # for i in range(2):
        #     self.currentPolicy.reinforcementLearning(input[i],output[i],result[i])
        if turn:  # turn이 True일때 백 학습
            self.trainNetwork.learning(input[0], output[0], result[0])
        else:  # 흑 학습
            self.trainNetwork.learning(input[1], output[1], result[1])

    def resettingPastPolicy(self):
        postPolicyFilePath = self.LaterAI.getNetwork().getFilePath()
        prePolicyFilePath = self.FormerAI.getNetwork().getFilePath()

        currentPolicyFileLists = os.listdir(postPolicyFilePath)

        # 학습된 체크포인트 파일을 pastPolicy디렉터리로 이동
        for filename in currentPolicyFileLists:
            previousPath = os.path.join(postPolicyFilePath, filename)
            afterPath = os.path.join(prePolicyFilePath, filename)
            shutil.copy(previousPath, afterPath)

        self.FormerAI.getNetwork().restoreCheckpoint()

    def playChessForReinforcementLearning(self, num):
        # 펜데이터로 경기가 끝날때 까지 진행
        chessBoard = chess.Board()
        turn=None

        if num % 2 == 0:
            white = self.FormerAI
            black = self.LaterAI
            turn=False
        else:
            white = self.LaterAI
            black = self.FormerAI
            turn =True
        gameCount = 0
        # for i in range(1000000000):
        #     pass

        while not chessBoard.is_game_over():
            objgraph.show_most_common_types()
            if gameCount >1000:
                break
            print("---------------")
            if chessBoard.turn:
                print("White Turn")
            else:
                print("Black Turn")
            print("a b c d e f g h")
            print("---------------")
            print(chessBoard, chr(13))
            print("---------------")
            print("a b c d e f g h")
            if chessBoard.turn:
                move = white.get_MCTS(chessBoard)
                # print("백: ", move)#, ", score: ", score)
            else:
                move = black.get_MCTS(chessBoard)
                # print("흑: ", move)#, ", score: ", score)
            chessBoard.push(chess.Move.from_uci(move))
            gameCount += 1

        if gameCount>1000:
            result = '1/2-1/2'
        else:
            result = chessBoard.result()

        if num % 2 == 0: #Later 흑, Former 백

            if result =='0-1': #Later Black 승
               self.gameInfo.info['CurrentLaterBlackWin']+=1
               self.gameInfo.info['LaterBlackWin']+=1
               self.gameInfo.info['LaterWin']+=1
               str = "LaterAi Win, Self-Play Result: " + result
            elif result =='1-0': #Later Black 패
                self.gameInfo.info['CurrentFormerWhiteWin'] += 1
                self.gameInfo.info['FormerWhiteWin']+=1
                self.gameInfo.info['FormerWin']+=1
                str = "LaterAi Lose, Self-Play Result: " + result
        else: # Later 백, Former 흑
            if result == '1-0':  # Later White 승
                self.gameInfo.info['CurrentLaterWhiteWin'] += 1
                self.gameInfo.info['LaterWhiteWin']+=1
                self.gameInfo.info['LaterWin']+=1
                str = "LaterAi Win, Self-Play Result: " + result
            elif result =='0-1': # Later White 패
                self.gameInfo.info['CurrentFormerBlackWin'] +=1
                self.gameInfo.info['FormerBlackWin']+1
                self.gameInfo.info['FormerWin']+=1
                str = "LaterAi Lose, Self-Play Result: " + result
        self.gameInfo.info['GameCount']+=1
        self.gameInfo.info['CurrentGameCount']+=1
        print(str)

        with open('Self-PlayResult.txt', 'a') as f:
            f.write(str + "\n")

        # 게임이 끝났을 때 체스 데이터를 저장
        fenDatas = self.saveRLData(chessBoard, result)
        self.reinforcementLearning(fenDatas,turn)

        #Game Info를 Json으로 저장
        self.gameInfo.save()
        isChange = self.gameInfo.is_ChangeCheckpoint()
        return isChange

    def fixResult(self, turn, result):
        if result == "1/2-1/2":
            return "draw"
        if turn:
            if result == "1-0":
                return "RL_Policy Win!"
            else:
                return "RL_Policy Lose!"
        else:
            if result == "1-0":
                return "RL_Policy Lose!"
            else:
                return "RL_Policy Win!"

if __name__ == '__main__':


    count = 0
    # for x in range(200):
    #     for y in range(10):
    #         sp = Play()
    #         sp.playChessForReinforcementLearning(count)
    #         count += 1
    #         del sp
    #     with open('../File/Self-PlayResult.txt', 'a') as f:
    #         f.write("-----------ResettingPastPolicy------------\n")
    #     sp.resettingPastPolicy()
    while True:
        sp = Play()
        isChange = sp.playChessForReinforcementLearning(count)
        count +=1

        if isChange:
            sp.resettingPastPolicy()
            sp.gameInfo.upRotaionCount()

        del sp

        with open('../File/Self-PlayResult.txt', 'a') as f:
            f.write("-----------Change Checkpoint------------\n")
