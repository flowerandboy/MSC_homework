import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from DataReader_0 import *
from HyperParameters_0 import *
from NeuralNet_0 import *

file_name = "housing.csv"

if __name__ == '__main__':
    reader = DataReader_0(file_name)
    reader.ReadData()
    reader.NormalizeX()
    reader.NormalizeY()

    hp = HyperParameters_0(13, 1, eta = 0.02, max_epoch = 200, batch_size = 10, eps = 1e-2 )
    net = NeuralNet_0(hp)
    net.train(reader, checkpoint = 0.1)


