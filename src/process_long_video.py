import cv2
import numpy as np
import mediapipe as mp
import os

class HandFeatureExtractor:
    """
    手部关键点提取与动作特征处理模块
    """
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,          # 连续帧优化
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hand_graph = self._build_hand_graph()  # 手部邻接矩阵

    def _build_hand_graph(self):
        """
        构建21个手部关键点的邻接矩阵，用于绘制连线
        """
        # 初始化21x21零矩阵
        graph = np.zeros((21, 21))
        # 示例：手腕连接手掌、手掌连接手指根部（实际可按MediaPipe官方拓扑填充）
        connections = [
            (0,1),(1,2),(2,3),(3,4),    # 拇指
            (0,5),(5,6),(6,7),(7,8),    # 食指
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
        提取视频中每帧的手部关键点轨迹
        返回列表，每个元素为{'right': np.array, 'left': np.array}或空字典
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

    def detect_action_boundaries(self, trajectory, window_size=5, step=4, stationary_thresh=0.01):
        """
        基于运动能量和滑动窗口检测动作边界
        """
        energies = []
        for idx, frame_data in enumerate(trajectory):
            if not frame_data:
                energies.append(0)
                continue
            # 计算当前帧与前一帧的速度
            if idx == 0:
                energies.append(0)
                continue
            prev_data = trajectory[idx-1]
            total_energy = 0
            for hand in ['left', 'right']:
                if hand in frame_data and hand in prev_data:
                    diff = frame_data[hand] - prev_data[hand]
                    speed = np.linalg.norm(diff, axis=1)
                    total_energy += np.sum(speed**2)
            energies.append(total_energy)

        # 滑动窗口计算平均能量
        window_energies = []
        for i in range(0, len(energies) - window_size + 1, step):
            window_energies.append(np.mean(energies[i:i+window_size]))

        # 识别静止段
        stationary_segments = []
        in_stationary = False
        for i, w_energy in enumerate(window_energies):
            if w_energy < stationary_thresh and not in_stationary:
                start = i * step
                in_stationary = True
            elif w_energy >= stationary_thresh and in_stationary:
                end = (i-1) * step + window_size
                stationary_segments.append((start, end))
                in_stationary = False
        # 合并短间隔静止段
        merged_segments = []
        for seg in stationary_segments:
            if not merged_segments:
                merged_segments.append(seg)
            else:
                last_seg = merged_segments[-1]
                if seg[0] - last_seg[1] < 3:
                    merged_segments[-1] = (last_seg[0], seg[1])
                else:
                    merged_segments.append(seg)
        # 根据静止段确定动作边界
        action_boundaries = []
        prev_end = 0
        for seg in merged_segments:
            if seg[0] - prev_end > 3:
                action_boundaries.append((prev_end, seg[0]-1))
            prev_end = seg[1]
        if prev_end < len(energies):
            action_boundaries.append((prev_end, len(energies)-1))
        return action_boundaries

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
        简单示例：提取统计特征
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
        根据动作边界生成带关键点标注的短视频
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
video_file = "../data/raw/video/long_hand_video_04.mp4"
output_folder = "../data/processed/video/segments_04"
os.makedirs(output_folder, exist_ok=True)

extractor = HandFeatureExtractor()

# 提取关键点轨迹
trajectory = extractor.process_video(video_file)

# 检测动作边界
boundaries = extractor.detect_action_boundaries(trajectory)

# 提取特征向量
features = extractor.extract_temporal_features(trajectory, boundaries)

# 生成标注视频
annotated_video_path = os.path.join(output_folder, "annotated_actions.mp4")
extractor.generate_annotated_video(video_file, trajectory, boundaries, annotated_video_path)

print(f"检测到动作段: {boundaries}")
print(f"特征向量 shape 示例: {[f.shape for f in features]}")
print(f"标注视频已保存到 {annotated_video_path}")
