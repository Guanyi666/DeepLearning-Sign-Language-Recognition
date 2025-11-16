from src.process_video.process_video import process_video
import numpy as np

input_video = "./data/raw/video/hand_video_test_01.mp4"

process_video(
    input_video,
    output_feature_path="./data/processed/video/example_features.npy",
    output_video_path="./data/processed/video/example_annotated.mp4",
    target_fps=5,
    window_size=30,
    stride=15
)

# 用stride < window_size来使窗口重叠，生成更多样本，提高训练数据数据量

test = np.load('./data/processed/video/example_features.npy')

print(test)
