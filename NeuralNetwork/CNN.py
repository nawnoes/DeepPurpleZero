"""
기존 알파고 Lee와 다른점
- L2 weight regulazation
- Policy와 Value Network를 결합
- 많은 Convolutional residual block 사용 
- 순전히 Self-Play를 통한 강화학습

"""
from abc import *
import chess
import numpy as np
import tensorflow as tf
import os
import copy
import Support.Board2Array as B2A
import Support.OneHotEncoding as OHE

learning_rate= 0.0001

class NeuralNetwork(metaclass=ABCMeta):
    def __init__(self,name,filePath):
        super().__init__()
        self.networkName= name
        self.networkFilePath = filePath
    @abstractmethod
    def model(self):
        pass
    @abstractmethod
    def getInput(self):
        pass
    @abstractmethod
    def getOutput(self):
        pass
    def getFilePath(self):
        return self.networkFilePath