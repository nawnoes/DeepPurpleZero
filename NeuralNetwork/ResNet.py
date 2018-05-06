import chess
import numpy as np
import tensorflow as tf
import os
import copy
import Support.Board2Array as B2A
from Support.OneHotEncoding import OneHotEncode as OHE

learning_rate= 0.001
beta = 0.001

class DeepPurpleNetwork:
    def __init__(self,filePath ="../Checkpoint/", is_traing =True):
        self.checkpointPath = filePath
        self.input = None
        self.model(is_traing)
        self.global_step = 0
        self.checkpointPath=filePath
        self.restore()

    def model(self, is_traing = True):
        self.X = tf.placeholder(tf.float32, [None, 8, 8, 35], name="X")  # 체스에서 8X8X10 이미지를 받기 위해 64
        self.Y = tf.placeholder(tf.float32, [None, 4096], name="Y")
        self.Z = tf.placeholder(tf.float32, [None, 1], name="Z")

        W1 = tf.get_variable("W1", shape=[5, 5, 35, 128],initializer=tf.contrib.layers.xavier_initializer())
        W1_Regulization = tf.nn.l2_loss(W1)
        conv_W1 = tf.nn.conv2d(self.X, W1, strides=[1, 1, 1, 1], padding='SAME')
        batch_W1 = self.batchNormalization(conv_W1,is_traing)
        L1 = tf.nn.relu(batch_W1)


        """---- reidual block 1----"""
        W2 = tf.get_variable("W2", shape=[3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
        W2_Regulization = tf.nn.l2_loss(W2)
        conv_W2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
        batch_W2 = self.batchNormalization(conv_W2,is_traing)
        L2 = tf.nn.relu(batch_W2)

        W3 = tf.get_variable("W3", shape=[3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
        W3_Regulization = tf.nn.l2_loss(W3)
        conv_W3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
        batch_W3 = self.batchNormalization(conv_W3,is_traing)
        residual_W3 = tf.add(batch_W3 , L1)
        L3 = tf.nn.relu(residual_W3)

        """---- reidual block 2----"""
        W4 = tf.get_variable("W4", shape=[3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
        W4_Regulization = tf.nn.l2_loss(W4)
        conv_W4 = tf.nn.conv2d(L3, W4, strides=[1, 1, 1, 1], padding='SAME')
        batch_W4 = self.batchNormalization(conv_W4,is_traing)
        L4 = tf.nn.relu(batch_W4)

        W5 = tf.get_variable("W5", shape=[3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
        W5_Regulization = tf.nn.l2_loss(W5)
        conv_W5 = tf.nn.conv2d(L4, W5, strides=[1, 1, 1, 1], padding='SAME')
        batch_W5 = self.batchNormalization(conv_W5,is_traing)
        residual_W5 = tf.add(batch_W5 , L3)
        L5 = tf.nn.relu(residual_W5)

        """---- reidual block 3----"""
        W6 = tf.get_variable("W6", shape=[3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
        W6_Regulization = tf.nn.l2_loss(W6)
        conv_W6 = tf.nn.conv2d(L5, W6, strides=[1, 1, 1, 1], padding='SAME')
        batch_W6 = self.batchNormalization(conv_W6,is_traing)
        L6 = tf.nn.relu(batch_W6)

        W7 = tf.get_variable("W7", shape=[3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
        W7_Regulization = tf.nn.l2_loss(W7)
        conv_W7 = tf.nn.conv2d(L6, W7, strides=[1, 1, 1, 1], padding='SAME')
        batch_W7 = self.batchNormalization(conv_W7,is_traing)
        residual_W7 = tf.add(batch_W7,L5)
        L7 = tf.nn.relu(residual_W7)

        """---- reidual block 4----"""
        W8 = tf.get_variable("W8", shape=[3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
        W8_Regulization = tf.nn.l2_loss(W8)
        conv_W8 = tf.nn.conv2d(L7, W8, strides=[1, 1, 1, 1], padding='SAME')
        batch_W8 = self.batchNormalization(conv_W8,is_traing)
        L8 = tf.nn.relu(batch_W8)

        W9 = tf.get_variable("W9", shape=[3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
        W9_Regulization = tf.nn.l2_loss(W9)
        conv_W9 = tf.nn.conv2d(L8, W9, strides=[1, 1, 1, 1], padding='SAME')
        batch_W9 = self.batchNormalization(conv_W9,is_traing)
        residual_W9 = tf.add(batch_W9 ,L7)
        L9 = tf.nn.relu(residual_W9)

        """Policy """
        P_W = tf.get_variable("P_W", shape=[1, 1, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
        P_W_Regulization = tf.nn.l2_loss(P_W)
        P_conv_W = tf.nn.conv2d(L9, P_W, strides=[1, 1, 1, 1], padding='SAME')
        P_batch_W = self.batchNormalization(P_conv_W,is_traing)
        P_L = tf.nn.relu(P_batch_W)

        P_FlatLayer = tf.reshape(P_L, [-1, 8 * 8 * 128])
        P_Flat_W = tf.get_variable("P_Flat_W", shape=[8 * 8 * 128, 4096],initializer=tf.contrib.layers.xavier_initializer())
        P_Flat_W_Regulization = tf.nn.l2_loss(P_Flat_W)
        # P_Flat_B = tf.get_variable("P_Flat_B", initializer=tf.random_normal([4096], stddev=0.01))
        self.P_hypothesis = tf.matmul(P_FlatLayer, P_Flat_W)


        """ Value"""

        V_W = tf.get_variable("V_W", shape=[1, 1, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
        V_W_Regulization = tf.nn.l2_loss(V_W)
        V_conv_W = tf.nn.conv2d(L9, V_W, strides=[1, 1, 1, 1], padding='SAME')
        V_batch_W = self.batchNormalization(V_conv_W,is_traing)
        V_L = tf.nn.relu(V_batch_W)

        V_FlatLayer = tf.reshape(V_L, [-1, 8 * 8 * 128])
        V_Flat_W = tf.get_variable("V_Flat_W", shape=[8 * 8 * 128, 1],initializer=tf.contrib.layers.xavier_initializer())
        V_Flat_W_Regulization = tf.nn.l2_loss(V_Flat_W)
        V_Flat_B = tf.get_variable("V_Flat_B", initializer=tf.random_normal([1], stddev=0.01))
        self.V_hypothesis = tf.tanh(tf.matmul(V_FlatLayer, V_Flat_W))


        """Cost with L2 Regularization """
        regularizer = W1_Regulization+W2_Regulization+W3_Regulization+W4_Regulization+W5_Regulization+W6_Regulization+\
                      W7_Regulization +W8_Regulization+W9_Regulization+P_W_Regulization+P_Flat_W_Regulization+\
                      V_W_Regulization+V_Flat_W_Regulization
        self.logit= tf.nn.softmax_cross_entropy_with_logits(logits=self.P_hypothesis, labels=self.Y)
        self.valueError = tf.square(self.Z - self.V_hypothesis)
        self.cost = tf.add(self.valueError,self.logit)
        self.cost = tf.reduce_mean(self.cost+ beta * regularizer)
        self.cost = tf.reduce_mean(self.cost)

        # self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.cost)
        pn_correct_prediction = tf.equal(tf.argmax(self.P_hypothesis, 1), tf.argmax(self.Y, 1))
        vn_correct_prediction = (2 - tf.abs(self.Y - self.V_hypothesis)) / 2

        pn_accuracy = tf.reduce_mean(tf.cast(pn_correct_prediction, tf.float32)) * 100
        vn_accuracy = tf.reduce_mean(tf.cast(vn_correct_prediction, tf.float32)) * 100

        # GPU 사용 옵션
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        config.gpu_options.allocator_type = "BFC"
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()


    def batchNormalization(self, bnInput, decay = 0.999, is_traing=True):

        epsilon = 1e-5
        gamma = tf.Variable(tf.ones(shape=[bnInput.get_shape()[-1]]))
        beta = tf.Variable(tf.zeros(shape=[bnInput.get_shape()[-1]]))

        populationMean =  tf.Variable(tf.zeros(shape=[bnInput.get_shape()[-1]]), trainable=False)
        populationVar = tf.Variable(tf.ones(shape=[bnInput.get_shape()[-1]]), trainable=False)

        if is_traing:
            batchMean, batchVar = tf.nn.moments(bnInput,[0, 1, 2],name='moments')
            trainMean = tf.assign(populationMean, populationMean *decay + batchMean *(1 - decay))
            trainVar = tf.assign(populationVar, populationVar * decay + batchVar * (1 - decay))

            with tf.control_dependencies([trainMean, trainVar]):
                return tf.nn.batch_normalization(bnInput,batchMean,batchVar,beta,gamma,epsilon)
        else:
            return tf.nn.batch_normalization(bnInput, populationMean, populationVar, beta, gamma, epsilon)
    def learning(self, input, policylabel, valuelabel):
        p, l, ve, c, o =self.sess.run([self.P_hypothesis,self.logit, self.valueError, self.cost, self.optimizer], feed_dict={self.X: input, self.Y: policylabel, self.Z: valuelabel})
        self.saveCheckpoint(self.checkpointPath,1)
        return p,l,ve,c
    def predict(self, input):
        input = self.getInput(input)
        policy, value = self.sess.run([self.P_hypothesis, self.V_hypothesis], feed_dict={self.X:input} )
        return policy, value
    def restore(self):
        self.ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.checkpointPath))


        if self.ckpt and self.ckpt.model_checkpoint_path:
            self.global_step = int(self.ckpt.model_checkpoint_path.rsplit('-', 1)[1])
            self.saver.restore(self.sess, self.ckpt.model_checkpoint_path)
            print("Checkpoint Path: ", self.ckpt.model_checkpoint_path)
            print("Checkpoint 로딩 완료")
        else:
            print("Checkpoint 로딩 실패")
    def getInput(self, chessBoard):
        #여러 값을 구할 때 input을 구하는 호출을 많이 사용하므로 수정 필요.
        input = []
        # input.append(B2A().board2arrayFor41Feature(chessBoard.getChessBoard(), chessBoard.getPremoves()))
        input.append(B2A.Board2Array().board2array(chessBoard))
        return input
    def getPolicy(self,input):
        # softmaxOfPolicy = tf.nn.softmax(self.P_hypothesis)
        input = self.getInput(input)
        return self.sess.run(self.P_hypothesis, feed_dict={self.X: input})
    def getValue(self,input):
        input = self.getInput(input)
        return self.sess.run(self.V_hypothesis, feed_dict={self.X: input})
    def getPolicyAndValue(self,chessBoard):
        input = self.getInput(chessBoard)
        softmaxOfPolicy = tf.nn.softmax(self.P_hypothesis)
        policy, value = self.sess.run([softmaxOfPolicy,self.V_hypothesis], feed_dict={self.X:input})
        array4096 = policy
        array4096 = np.array(array4096[0])
        argmaxOfSoftmax = (-array4096).argsort()
        return array4096, argmaxOfSoftmax, value[0][0]
    def saveCheckpoint(self, filePath, batch_size):
        self.global_step += batch_size
        self.saver.save(self.sess, filePath, global_step=self.global_step)
    def getArraysOfPolicyNetwork(self,chessBoard):
        input = self.getInput(chessBoard)
        array4096 = self.getPolicy(input)
        array4096 = np.array(array4096[0])
        ArgmaxOfSoftmax = (-array4096).argsort()
        #내림차순으로 분류한 것을 리스트로 반환 받는다
        #ArgMaxOfSoftmax 크기별로 Index만 저장 되어있다. 0~4095
        #계산된 softmax값과
        # 크기별로 정렬된 index가 들어 있는 ArgMaxOfSoftmax 반환
        return array4096, ArgmaxOfSoftmax
    def getMove(self,chessBoard):
        #chessBoard가 ChessBoard()인지 chess.ChessBoard()인지에 따라 legalMoves 호출 방법이 변경
        #추후 변경 필요
        softMax = self.getPolicy(chessBoard)
        softMax = np.array(softMax[0])
        ArgMaxOfSoftmax = (-softMax).argsort()
        # 내림차순으로 분류한 것을 리스트로 반환 받는다
        # softMAxArgMax는 크기별로 Index만 저장 되어있다. 0~4095
        ohe = OHE()
        i = 0
        child = 0
        numOfLegalMoves = len(chessBoard.legal_moves)
        numOfChild = 1

        # 만드려고 하는 자식 개수보다 가능한 move 갯수가 적을때
        if numOfLegalMoves < numOfChild:
            numOfChild = numOfLegalMoves

        for j in range(4096):
            if child >= numOfChild:  # 만드려고 하는 자식 갯수보다 많으면 반환
                break
            try:
                tmpMove = ohe.indexToMove4096(ArgMaxOfSoftmax[i])
                strMove = copy.deepcopy(tmpMove)
                tmpMove = chess.Move.from_uci(tmpMove)  # 주석처리: 선피쉬랑 붙기 위해 String 자체를 사용
            except:
                None
            if tmpMove in chessBoard.legal_moves:
                move = strMove  # tmpMove가 legal이면 추가
                score = softMax[ArgMaxOfSoftmax[i]]
                print(i + 1, "번째 선택된 점수 : ", score, " move: ", move)
                child += 1
            else:
                strMove = strMove + "q"
                tmpMove = chess.Move.from_uci(strMove)
                if tmpMove in chessBoard.legal_moves:
                    move = strMove  # +"q"  # tmpMove가 legal이면 추가
                    score = softMax[ArgMaxOfSoftmax[i]]
                    print(i + 1, "번째 선택된 점수 : ", score, " move: ", move)
                    child += 1
            i += 1
        return move
    def deleteAllCheckpoint(self):
        filenames = os.listdir(self.checkpointPath)
        for filename in filenames:
            fullFilePath = os.path.join(self.checkpointPath, filename)
            os.remove(fullFilePath)
if __name__ == '__main__':
    g1 = tf.Graph()
    g2 = tf.Graph()
    with g1.as_default():
        rn = DeepPurpleNetwork()
    with g2.as_default():
        pNet = DeepPurpleNetwork(False)

    board = chess.Board()
    ohe = OHE()

    label = ohe.uciMoveToOnehot(ohe.indexToMove4096(0))

    # a,b,v = rn.getPolicyAndValue(board)
    # print("array4096: ", a)
    # print("argmaxOfSoftmax: ", b)
    # print("value: ", v)
    # gp = rn.getPolicy(board)
    # print("gp: ", gp)
    # gv = rn.getValue(board)
    # print("gv: ", gv)
    # gm = rn.getMove(board)
    # print("gm: ", gm)
    aInput=[]
    aInput.append(label)
    rn.restore()
    pNet.restore()
    p,v = pNet.predict(board)
    rp, rv = rn.predict(board)

    print(p)
    print(v)
    print(rp)
    print(rv)

    # for i in range(10000):
    #     p, l, ve, c = rn.learning(board, aInput,[[-1]])
    #     print("logit: ", l ," ve: ",ve, " c: ",c)
        # print("p: ",p)
        # print("Y: ",aInput)
        # a, b, v = rn.getPolicyAndValue(board)
        # print("array4096: ", a)
        # print("argmaxOfSoftmax: ", b)
        # print("value: ", v)
        # print("----------------------")
