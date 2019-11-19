import numpy as np
from pathlib import Path #路径库
import pandas as pd

file_name = "housing.csv"
data = pd.read_csv(file_name,header=None)
data1=np.array(data)
XRaw=data1[:,0:13]
YRaw=data1[:,13]
YRaw = YRaw.reshape(-1,1)
num_train = XRaw.shape[0]
#print(data)
print("data1:",data1)
print("XRaw:",XRaw)
print("XRaw[0]:",XRaw[0])
print("YRaw1:",YRaw)
print("样本数据数量:",num_train)
print("XRaw[0]:",XRaw[0])

seed = np.random.randint(0,100)
np.random.seed(seed)
print(seed)
