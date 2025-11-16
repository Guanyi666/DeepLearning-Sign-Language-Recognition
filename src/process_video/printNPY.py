import numpy as np

# 设置numpy打印选项，取消截断
np.set_printoptions(threshold=np.inf)

data = np.load('../../data/dataset/train/he/02/he_1_features.npy')
print(data.shape)
print(data)