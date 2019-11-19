import numpy as np
import pandas as pd
from pathlib import Path

class DataReader_0():
    def __init__(self, data_file):
        self.train_file_name = data_file #存储文件名
        self.num_train = 0
        self.XTrain = None #存储x值到XTrain数组中
        self.YTrain = None #存储标签值到YTrain数组中
        self.XRaw=None
        self.YRaw=None

    # 从文件读数据
    def ReadData(self):
        train_file = Path(self.train_file_name)
        if train_file.exists():
            #加载数据到data中
            data = pd.read_csv(self.train_file_name,header=None)
            data1 = np.array(data)#生成506×14矩阵,包括所有x和y
            self.XRaw=data1[:,0:13]#获取样本的矩阵，排除y
            YRaw = data1[:,13]#获取所有y值
            self.YRaw = YRaw.reshape(-1,1)#将所有y值排成506×1矩阵

            self.XTrain=self.XRaw
            self.YTrain=self.YRaw
            self.num_train=self.XRaw.shape[0]

        else:
            raise Exception("Cannot find train file!!!")

    #训练数据归一化
    def NormalizeX(self):
        X_new = np.zeros(self.XRaw.shape) #创建X_new矩阵存储归一化后的训练数据
        num_feature = self.XRaw.shape[1] #变量种类（列数）
        self.X_norm = np.zeros((num_feature, 2))#对每个变量都存储两个值

        for i in range(num_feature):
            col_i = self.XRaw[:,i]#获取列
            self.X_norm[i, 0] = np.min(col_i)#使用self.在其他方法中还要使用
            #print("X_norm[i,0]",self.X_norm[i, 0])
            self.X_norm[i, 1] = np.max(col_i) - np.min(col_i)
            #第i列数据更新：
            X_new[:,i] = (col_i - self.X_norm[i, 0]) / self.X_norm[i, 1]
        #得到了符合训练要求的数据
        self.XTrain = X_new

    #标签值初始化
    def NormalizeY(self):
        self.Y_norm = np.zeros((1, 2))
        self.Y_norm[0, 0] = np.min(self.YRaw)
        self.Y_norm[0, 1] = np.max(self.YRaw) - np.min(self.YRaw)

        Y_new = (self.YRaw - self.Y_norm[0,0]) / self.Y_norm[0,1]
        self.YTrain = Y_new
        #print("YTrain:",self.YTrain)

    #待预测数据初始化
    def NormalizePredictateData(self, x):
        X_new = np.zeros(x.shape)#存储修正后的待预测数据
        n = X_new.shape[1]
        for i in range(n):
            X_new[:, i] = (x[:, i] - self.X_norm[i, 0]) / self.X_norm[i, 1]
        return X_new

    # 获取单个样本
    def GetSingleTrainSample(self, iteration):
        x = self.XTrain[iteration]
        y = self.YTrain[iteration]
        return x, y

    # 一次迭代获取的样本
    def GetBatchTrainSamples(self, batch_size, iteration):
        start = iteration * batch_size
        end = start + batch_size
        # 分批次取数据
        batch_X = self.XTrain[start:end,:]
        batch_Y = self.YTrain[start:end,:]
        return batch_X, batch_Y

    def GetWholeTrainSamples(self):
        return self.XTrain, self.YTrain

    def Shuffle(self):
        seed = np.random.randint(0,100)
        np.random.seed(seed)

        #打乱self.XTrain矩阵的行，每行元素不变
        XP = np.random.permutation(self.XTrain)
        np.random.seed(seed)
        #打乱self.YTrain矩阵的行
        YP = np.random.permutation(self.YTrain)

        #更新self.XTrain和self.YTrain
        # 打乱后XP，YP的行和打乱前是对应的
        self.XTrain = XP
        self.YTrain = YP
