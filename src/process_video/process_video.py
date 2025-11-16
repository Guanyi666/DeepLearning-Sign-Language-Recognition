import cv2
import numpy as np
from src.feature_extractor import SingleHandFeatureExtractor


def process_video(
    video_path,
    output_feature_path,
    output_video_path=None,
    target_fps=5,
    window_size=30,
    stride=15
):
    extractor = SingleHandFeatureExtractor()
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否打开成功
    if not cap.isOpened():
        print("❌ Error: 无法打开视频，请检查视频路径或格式：", video_path)
        return

    # 获取原 FPS
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps == 0 or original_fps is None:
        print("⚠ Warning: 原视频 FPS 读取失败，将使用默认 FPS = 30")
        original_fps = 30

    print("原始 FPS =", original_fps)

    # 防止 frame_interval = 0
    frame_interval = max(int(round(original_fps / target_fps)), 1)
    print("帧采样间隔 =", frame_interval)

    features = []
    sampled_frames = []

    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("视频总帧数 =", total_frames)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            feature_vec = extractor.process_frame(frame)
            features.append(feature_vec)
            sampled_frames.append(frame)

        frame_idx += 1

    cap.release()

    print("共抽取帧数:", len(sampled_frames))
    features = np.array(features)
    print("单帧特征 shape =", features.shape)

    # ========= 滑动窗口 =========
    windows = []
    for start in range(0, len(features), stride):
        window = features[start:start + window_size]

        # 如果窗口不足 window_size，使用 0 向量补齐
        if len(window) < window_size:
            pad = np.zeros((window_size - len(window), features.shape[1]))
            window = np.vstack([window, pad])

        windows.append(window)

    windows = np.array(windows)

    print("===================================")
    np.save(output_feature_path, windows)
    print(f"特征向量序列保存到: {output_feature_path}")
    print("窗口切片 shape =", windows.shape)
    print("===================================")

    # ========= 关键点视频 =========
    if output_video_path and len(sampled_frames) > 0:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        h, w, _ = sampled_frames[0].shape
        out = cv2.VideoWriter(output_video_path, fourcc, target_fps, (w, h))

        for frame in sampled_frames:
            annotated_frame = extractor.draw_landmarks(frame.copy())
            out.write(annotated_frame)

        out.release()
        print(f"关键点标注视频保存到: {output_video_path}")
