import numpy as np

# 设置numpy打印选项，取消截断
np.set_printoptions(threshold=np.inf)

data = np.load('../../data/processed/video/repeat_video_01/action_2_features.npy')
print(data.shape)
print(data)