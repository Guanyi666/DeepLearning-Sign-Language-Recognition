import cv2
import numpy as np
import mediapipe as mp
import os
from src.feature_extractor import SingleHandFeatureExtractor

class HandFeatureExtractor:
    """
    手部关键点提取与动作特征处理模块（优化重复动作分割）
    """
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hand_graph = self._build_hand_graph()

    def _build_hand_graph(self):
        """
        构建21个手部关键点的邻接矩阵
        """
        graph = np.zeros((21, 21))
        connections = [
            (0,1),(1,2),(2,3),(3,4),      # 拇指
            (0,5),(5,6),(6,7),(7,8),      # 食指
            (0,9),(9,10),(10,11),(11,12), # 中指
            (0,13),(13,14),(14,15),(15,16), # 无名指
            (0,17),(17,18),(18,19),(19,20)  # 小指
        ]
        for i,j in connections:
            graph[i,j] = 1
            graph[j,i] = 1
        return graph

    def process_video(self, video_path):
        """
        提取视频中每帧手部关键点轨迹
        """
        cap = cv2.VideoCapture(video_path)
        trajectory = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)
            frame_data = self._process_results(results)
            trajectory.append(frame_data)
        cap.release()
        return trajectory

    def _process_results(self, results):
        """
        处理MediaPipe结果，返回左右手关键点
        """
        data = {}
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_type = handedness.classification[0].label  # 'Left' 或 'Right'
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                data[hand_type.lower()] = landmarks
        return data

    def detect_action_boundaries_repeat(self, trajectory,
                                            min_action_len=5,
                                            max_pause_len=25):
        """
        允许动作中有短暂停留，避免将一个动作拆成多个
        """
        energies = []
        hands_present = []

        for idx, frame_data in enumerate(trajectory):
            has_hand = any(hand in frame_data for hand in ['left', 'right'])
            hands_present.append(has_hand)

            if idx == 0 or not frame_data or not trajectory[idx - 1]:
                energies.append(0)
                continue

            prev_data = trajectory[idx - 1]
            total_energy = 0
            for hand in ['left', 'right']:
                if hand in frame_data and hand in prev_data:
                    diff = frame_data[hand] - prev_data[hand]
                    speed = np.linalg.norm(diff, axis=1)
                    total_energy += np.sum(speed ** 2)
            energies.append(total_energy)

        energies = np.array(energies)
        # 简单平滑
        window_size = 3
        smooth_energies = np.convolve(energies, np.ones(window_size) / window_size, mode='same')
        energy_thresh = np.mean(smooth_energies) * 0.5 + 1e-6

        boundaries = []
        in_action = False
        pause_counter = 0

        for i in range(len(trajectory)):
            e = smooth_energies[i]
            hand_now = hands_present[i]

            # 动作开始条件
            if not in_action and (hand_now or e > energy_thresh):
                start = i
                in_action = True
                pause_counter = 0

            # 动作进行中
            if in_action:
                if not hand_now or e < energy_thresh:
                    pause_counter += 1
                else:
                    pause_counter = 0

                # 动作结束条件：连续静止帧数超过阈值
                if pause_counter >= max_pause_len:
                    end = i - pause_counter
                    if end - start + 1 >= min_action_len:
                        boundaries.append((start, end))
                    in_action = False
                    pause_counter = 0

        # 视频结束时仍在动作中
        if in_action:
            boundaries.append((start, len(trajectory) - 1))

        return boundaries

    def extract_temporal_features(self, trajectory, boundaries, method='stats'):
        """
        从每个动作区间提取时序特征
        """
        features = []
        for start, end in boundaries:
            action_traj = trajectory[start:end+1]
            if method == 'stats':
                feat = self.calculate_stats_features(action_traj)
            elif method == 'stgcn':
                feat = self.convert_to_stgcn_input(action_traj)
            features.append(feat)
        return features

    def calculate_stats_features(self, action_traj):
        """
        提取统计特征（均值+方差）
        """
        all_feats = []
        for frame in action_traj:
            frame_vec = []
            for hand in ['left','right']:
                if hand in frame:
                    coords = frame[hand]
                    mean = np.mean(coords, axis=0)
                    std = np.std(coords, axis=0)
                    frame_vec.extend(mean.tolist() + std.tolist())
                else:
                    frame_vec.extend([0]*6)
            all_feats.append(frame_vec)
        return np.array(all_feats)

    def generate_annotated_video(self, video_path, trajectory, boundaries, output_path):
        """
        生成带关键点标注的动作视频
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w,h))

        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx < len(trajectory):
                frame_data = trajectory[frame_idx]
                if frame_data:
                    for hand, hand_data in frame_data.items():
                        color = (0,0,255) if hand=='right' else (255,0,0)
                        for lm in hand_data:
                            cv2.circle(frame, (int(lm[0]*w), int(lm[1]*h)), 5, color, -1)
                        for i in range(21):
                            for j in range(21):
                                if self.hand_graph[i][j]==1:
                                    cv2.line(frame,
                                             (int(hand_data[i][0]*w), int(hand_data[i][1]*h)),
                                             (int(hand_data[j][0]*w), int(hand_data[j][1]*h)),
                                             color, 2)
            for i, (start,end) in enumerate(boundaries):
                if frame_idx >= start and frame_idx <= end:
                    cv2.putText(frame, f'Action {i+1}', (10,30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    break
            out.write(frame)
            frame_idx +=1
        out.release()

# ----------------------------
# 使用示例
# ----------------------------
video_file = "../../data/raw/thank_you/thank_you_03.mp4"
output_folder = "../../data/dataset/train/thank_you/03"
os.makedirs(output_folder, exist_ok=True)

extractor = HandFeatureExtractor()

# 提取关键点轨迹
trajectory = extractor.process_video(video_file)

# 检测动作边界
boundaries = extractor.detect_action_boundaries_repeat(trajectory)

# 提取特征向量
features = extractor.extract_temporal_features(trajectory, boundaries)

# 生成标注视频
annotated_video_path = os.path.join(output_folder, "annotated_actions.mp4")
extractor.generate_annotated_video(video_file, trajectory, boundaries, annotated_video_path)

def extract_action_features(video_path, boundaries, output_dir, fixed_frame_count=30):
    """
    从视频中提取每个动作的多维特征向量，并保存到指定目录
    video_path: 视频文件路径
    boundaries: 动作边界列表 [(start_frame, end_frame), ...]
    output_dir: 特征向量保存目录
    fixed_frame_count: 每个动作固定抽取帧数
    """
    os.makedirs(output_dir, exist_ok=True)
    extractor = SingleHandFeatureExtractor()

    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Error: 无法打开视频", video_path)
        return

    # 读取全部帧
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    print(f"视频总帧数: {len(frames)}")
    all_action_features = []

    for idx, (start, end) in enumerate(boundaries):
        action_frames = frames[start:end+1]

        # 如果动作帧数少于固定帧数，则重复最后一帧补齐
        if len(action_frames) < fixed_frame_count:
            repeats = fixed_frame_count - len(action_frames)
            action_frames += [action_frames[-1]] * repeats
        # 如果动作帧数多于固定帧数，则等间隔采样
        elif len(action_frames) > fixed_frame_count:
            indices = np.linspace(0, len(action_frames)-1, fixed_frame_count, dtype=int)
            action_frames = [action_frames[i] for i in indices]

        # 提取特征
        features = np.array([extractor.process_frame(f) for f in action_frames])
        all_action_features.append(features)

        # 保存每个动作
        save_path = os.path.join(output_dir, f"thank_you_{idx+1}_features.npy")
        np.save(save_path, features)
        print(f"动作 {idx+1} 特征 shape: {features.shape}, 保存到: {save_path}")

    all_action_features = np.array(all_action_features)  # shape: (动作数, fixed_frame_count, 特征维度)
    print(f"全部动作特征 shape: {all_action_features.shape}")
    return all_action_features


extract_action_features(
    video_file,
    boundaries=boundaries,
    output_dir=output_folder,
    fixed_frame_count=30
)

print(f"检测到动作段: {boundaries}")
print(f"特征向量 shape 示例: {[f.shape for f in features]}")
print(f"标注视频已保存到 {annotated_video_path}")