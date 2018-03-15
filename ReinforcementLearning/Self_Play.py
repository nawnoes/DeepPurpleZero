import ReinforcementLearning.ChessAI as AI
from  NeuralNetwork.ResNet import DeepPurpleNetwork as DPN
import chess
import copy
import Support.FenLoad as FL
import os
import shutil
import tensorflow as tf
# import ChessBoard

class Play:
    def __init__(self):
        g1 = tf.Graph()
        g2 = tf.Graph()
        g3 = tf.Graph()
        postCheckpointPath = '../Checkpoint/PostCheckpoint/'
        preCheckpointPath = '../Checkpoint/PreCheckpoint/'
        trainCheckpointPath = postCheckpointPath
        with g1.as_default():
            self.postAI = AI.ChessAI(postCheckpointPath)
        with g2.as_default():
            self.preAI = AI.ChessAI(preCheckpointPath)
        with g3.as_default(): #강화학습을 위한
            self.trainNetwork = DPN(trainCheckpointPath)

        self.loadFenData= FL.FenLoad()

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
            self.trainNetwork.reinforcementLearning(input[0], output[0], result[0])
        else:  # 흑 학습
            self.trainNetwork.reinforcementLearning(input[1], output[1], result[1])

    def resettingPastPolicy(self):
        postPolicyFilePath = self.postAI.getNetwork().getFilePath()
        prePolicyFilePath = self.preAI.getNetwork().getFilePath()

        currentPolicyFileLists = os.listdir(postPolicyFilePath)

        # 학습된 체크포인트 파일을 pastPolicy디렉터리로 이동
        for filename in currentPolicyFileLists:
            previousPath = os.path.join(postPolicyFilePath, filename)
            afterPath = os.path.join(prePolicyFilePath, filename)
            shutil.copy(previousPath, afterPath)

        self.preAI.getNetwork().restoreCheckpoint()

    def playChessForReinforcementLearning(self, num):
        # 펜데이터로 경기가 끝날때 까지 진행
        chessBoard = chess.Board()

        if num % 2 == 0:
            white = self.preAI
            black = self.postAI
        else:
            white = self.postAI
            black = self.preAI

        gameCount = 0

        while not chessBoard.is_game_over():
            if gameCount >100:
                break
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

        if gameCount>100:
            result = '1/2-1/2'
        else:
            result = chessBoard.result()

        if num % 2 == 0:
            str = "currentPolicy: Black/ Self-Play Result: " + result
        else:
            str = "currentPolicy: White/ Self-Play Result: " + result
        print(str)

        with open('Self-PlayResult.txt', 'a') as f:
            f.write(str + "\n")

        # 게임이 끝났을 때 체스 데이터를 저장
        fenDatas = self.saveRLData(chessBoard, result)
        self.reinforcementLearning(fenDatas)

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
    sp = Play()

    count = 0
    for x in range(200):
        for y in range(10):
            sp.playChessForReinforcementLearning(count)
            count += 1
        with open('Self-PlayResult.txt', 'a') as f:
            f.write("-----------ResettingPastPolicy------------\n")
        sp.resettingPastPolicy()