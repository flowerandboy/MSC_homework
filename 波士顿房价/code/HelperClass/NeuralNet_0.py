import numpy as np
import matplotlib.pyplot as plt
import math
from pathlib import Path
from matplotlib.colors import LogNorm

from DataReader_0 import *
from HyperParameters_0 import *

class NeuralNet_0():
    def __init__(self,hp):
        self.hp = hp #self.hp设为一个类
        self.W = np.zeros((self.hp.input_size,self.hp.output_size))
        self.B = np.zeros((1,self.hp.output_size))

    def __forwardBatch(self, batch_x):
        Z = np.dot(batch_x, self.W) + self.B
        return Z

    def __backwardBatch(self, batch_x, batch_y, batch_z):
        m = batch_x.shape[0]
        dZ = batch_z - batch_y
        dB = dZ.sum(axis = 0, keepdims = True)
        dW = np.dot(batch_x.T, dZ) / m
        return dW, dB

    def __updata(self, dW, dB):
        self.W = self.W - dW * self.hp.eta
        self.B = self.B - dB * self.hp.eta

    def inference(self, x):
        return self.__forwardBatch(x)

    def train(self, dataReader, checkpoint = 0.1):
        loss = 10
        if self.hp.batch_size == -1:
            self.hp.batch_size = dataReader.num_train
        max_iteration = math.ceil(dataReader.num_train / self.hp.batch_size)
        checkpoint_iteration = int(max_iteration * checkpoint)

        for epoch in range(self.hp.max_epoch):
            print("epoch=%d" %epoch)
            dataReader.Shuffle()#每次训练打乱数据顺序
            for iteration in range(max_iteration):
                #更新w和b
                batch_x, batch_y = dataReader.GetBatchTrainSamples(self.hp.batch_size, iteration)
                batch_z = self.__forwardBatch(batch_x)
                dW, dB = self.__backwardBatch(batch_x, batch_y, batch_z)
                self.__updata(dW, dB)
                total_iteration = epoch * max_iteration + iteration#总的迭代次数
                if (total_iteration + 1) % checkpoint_iteration == 0:
                    loss = self.checkLoss(dataReader)
                    print(epoch, iteration, loss, self.W, self.B)
                    if loss < self.hp.eps:
                        break #退出内层循环
            if loss < self.hp.eps:
                break
        print("W=", self.W)
        print("B=", self.B)

    def checkLoss(self, dataReader):
        X,Y = dataReader.GetWholeTrainSamples()
        m = X.shape[0]
        Z = self.__forwardBatch(X)
        LOSS = (Z - Y)**2
        loss = LOSS.sum()/(m * 2)
        return loss




