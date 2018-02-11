import chess
import numpy as np
import tensorflow as tf
import os
import copy
import Support.Board2Array as B2A
import Support.OneHotEncoding as OHE

learning_rate= 0.0001

class Network:
    def __init__(self):
        pass

    def model(self, is_traing):
        X = tf.placeholder(tf.float32, [None, 8, 8, 41], name="X")  # 체스에서 8X8X10 이미지를 받기 위해 64
        Y = tf.placeholder(tf.float32, [None, 4096], name="Y")

        W1 = tf.get_variable("W1", shape=[5, 5, 41, 128],initializer=tf.contrib.layers.xavier_initializer())
        conv_W1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
        batch_W1 = self.batchNormalization(conv_W1)
        L1 = tf.nn.relu(batch_W1)


        """---- reidual block 1----"""
        W2 = tf.get_variable("W2", shape=[3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
        conv_W2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
        batch_W2 = self.batchNormalization(conv_W2)
        L2 = tf.nn.relu(batch_W2)

        W3 = tf.get_variable("W3", shape=[3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
        conv_W3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
        batch_W3 = self.batchNormalization(conv_W3)
        residual_W3 = tf.add(batch_W3 , L1)
        L3 = tf.nn.relu(residual_W3)

        """---- reidual block 2----"""
        W4 = tf.get_variable("W4", shape=[3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
        conv_W4 = tf.nn.conv2d(L3, W4, strides=[1, 1, 1, 1], padding='SAME')
        batch_W4 = self.batchNormalization(conv_W4)
        L4 = tf.nn.relu(batch_W4)

        W5 = tf.get_variable("W5", shape=[3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
        conv_W5 = tf.nn.conv2d(L4, W5, strides=[1, 1, 1, 1], padding='SAME')
        batch_W5 = self.batchNormalization(conv_W5)
        residual_W5 = tf.add(batch_W5 , L3)
        L5 = tf.nn.relu(residual_W5)

        """---- reidual block 3----"""
        W6 = tf.get_variable("W6", shape=[3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
        conv_W6 = tf.nn.conv2d(L5, W6, strides=[1, 1, 1, 1], padding='SAME')
        batch_W6 = self.batchNormalization(conv_W6)
        L6 = tf.nn.relu(batch_W6)

        W7 = tf.get_variable("W7", shape=[3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
        conv_W7 = tf.nn.conv2d(L6, W7, strides=[1, 1, 1, 1], padding='SAME')
        batch_W7 = self.batchNormalization(conv_W7)
        residual_W7 = tf.add(batch_W7,L5)
        L7 = tf.nn.relu(residual_W7)

        """---- reidual block 4----"""
        W8 = tf.get_variable("W8", shape=[3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
        conv_W8 = tf.nn.conv2d(L7, W8, strides=[1, 1, 1, 1], padding='SAME')
        batch_W8 = self.batchNormalization(conv_W8)
        L8 = tf.nn.relu(batch_W8)

        W9 = tf.get_variable("W9", shape=[3, 3, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
        conv_W9 = tf.nn.conv2d(L8, W9, strides=[1, 1, 1, 1], padding='SAME')
        batch_W9 = self.batchNormalization(conv_W9)
        residual_W9 = tf.add(batch_W9 ,L7)
        L9 = tf.nn.relu(residual_W9)

        """Policy """
        P_W = tf.get_variable("P_W", shape=[1, 1, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
        P_conv_W = tf.nn.conv2d(L9, P_W, strides=[1, 1, 1, 1], padding='SAME')
        P_batch_W = self.batchNormalization(P_conv_W)
        P_L = tf.nn.relu(P_batch_W)

        P_FlatLayer = tf.reshape(P_L, [-1, 8 * 8 * 128])
        P_Flat_W = tf.get_variable("P_Flat_W", shape=[8 * 8 * 128, 4096],initializer=tf.contrib.layers.xavier_initializer())
        P_Flat_B = tf.get_variable("P_Flat_B", initializer=tf.random_normal([4096], stddev=0.01))
        P_hypothesis = tf.matmul(P_FlatLayer, P_Flat_W) + P_Flat_B


        """ Value"""

        V_W = tf.get_variable("V_W", shape=[1, 1, 128, 128], initializer=tf.contrib.layers.xavier_initializer())
        V_conv_W = tf.nn.conv2d(L9, V_W, strides=[1, 1, 1, 1], padding='SAME')
        V_batch_W = self.batchNormalization(V_conv_W)
        V_L = tf.nn.relu(V_batch_W)

        V_FlatLayer = tf.reshape(V_L, [-1, 8 * 8 * 128])
        V_Flat_W = tf.get_variable("V_Flat_W", shape=[8 * 8 * 128, 4096],initializer=tf.contrib.layers.xavier_initializer())
        V_Flat_B = tf.get_variable("V_Flat_B", initializer=tf.random_normal([4096], stddev=0.01))
        V_hypothesis = tf.tanh(tf.matmul(V_FlatLayer, V_Flat_W) + V_Flat_B)

        return P_hypothesis,V_hypothesis

    def batchNormalization(self, bnInput, decay = 0.999, is_traing=True):
        epsilon = 1e-7
        gamma = tf.Variable(tf.ones(shape=[bnInput.get_shape()[-1]]))
        beta = tf.Variable(tf.zores(shape=[bnInput.get_shape()[-1]]))

        populationMean =  tf.Variable(tf.zores(shape=[bnInput.get_shape()[-1]]), trainable=False)
        populationVar = tf.Variable(tf.ones(shape=[bnInput.get_shape()[-1]]), trainable=False)

        if is_traing:
            batchMean, batchVar = tf.nn.moments(bnInput,[0, 1, 2],name='moments')
            trainMean = tf.assign(populationMean, populationMean *decay + batchMean *(1 - decay))
            trainVar = tf.assign(populationVar, populationVar * decay + batchVar * (1 - decay))

            with tf.control_dependencies([trainMean, trainVar]):
                return tf.nn.batch_normalization(bnInput,batchMean,batchVar,beta,gamma,epsilon)
        else:
            #batch normalization 식 계산
            return tf.nn.batch_normalization(bnInput, populationMean, populationVar, beta, gamma, epsilon)

    def restore(self):
    def get
