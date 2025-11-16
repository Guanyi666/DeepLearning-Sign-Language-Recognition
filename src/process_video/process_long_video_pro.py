import cv2
import mediapipe as mp
import numpy as np
import json
import os
from typing import List, Tuple, Dict, Any
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SignLanguageActionSegmenter:
    """
    MediaPipe手语动作分割与特征提取系统
    """

    def __init__(self, detection_confidence: float = 0.5, tracking_confidence: float = 0.5):
        """
        初始化系统参数

        Args:
            detection_confidence: 检测置信度阈值
            tracking_confidence: 跟踪置信度阈值
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # 初始化MediaPipe姿态估计器
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )

        # 配置参数
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        self.segmentation_threshold = 0.2  # 动作分割阈值
        self.min_action_frames = 5  # 最小动作帧数
        self.window_size = 5  # 滑动窗口大小

        # 关键关节定义（手部相关）
        self.hand_joints = [
            [11, 13, 15],  # 左肩-左肘-左手腕
            [12, 14, 16],  # 右肩-右肘-右手腕
            [13, 15, 17],  # 左肘-左手腕-左手
            [14, 16, 18],  # 右肘-右手腕-右手
            [11, 12, 13],  # 左肩-右肩-左肘
            [12, 11, 14],  # 右肩-左肩-右肘
        ]

        # 手部关键点索引
        self.hand_indices = [15, 16, 17, 18, 19, 20, 21, 22]  # 左右手腕和手指关键点

    def extract_landmarks(self, video_path: str) -> List:
        """
        提取视频中每一帧的关键点

        Args:
            video_path: 输入视频路径

        Returns:
            关键点列表，每个元素对应一帧的关键点数据
        """
        logger.info(f"开始提取视频 {video_path} 的关键点...")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        landmarks_list = []
        frame_count = 0

        try:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    break

                # 转换图像格式
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False

                # 处理图像并提取关键点
                results = self.pose.process(image_rgb)

                # 存储关键点数据
                if results.pose_landmarks:
                    landmarks_list.append(results.pose_landmarks.landmark)
                else:
                    # 如果未检测到关键点，添加空值
                    landmarks_list.append(None)

                frame_count += 1
                if frame_count % 100 == 0:
                    logger.info(f"已处理 {frame_count} 帧")

        except Exception as e:
            logger.error(f"提取关键点时发生错误: {e}")
            raise
        finally:
            cap.release()

        logger.info(f"关键点提取完成，共处理 {frame_count} 帧")
        return landmarks_list

    def calculate_joint_angles(self, landmarks) -> List[float]:
        """
        计算关节角度

        Args:
            landmarks: 关键点数据

        Returns:
            关节角度列表
        """
        if not landmarks:
            return []

        angles = []
        for joint_triplet in self.hand_joints:
            try:
                # 获取三个点的坐标
                p1 = np.array([landmarks[joint_triplet[0]].x, landmarks[joint_triplet[0]].y])
                p2 = np.array([landmarks[joint_triplet[1]].x, landmarks[joint_triplet[1]].y])
                p3 = np.array([landmarks[joint_triplet[2]].x, landmarks[joint_triplet[2]].y])

                # 计算向量
                v1 = p1 - p2
                v2 = p3 - p2

                # 计算角度
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 防止数值误差
                angle = np.degrees(np.arccos(cos_angle))
                angles.append(angle)
            except (IndexError, AttributeError):
                # 如果关键点不存在，添加默认值
                angles.append(0.0)

        return angles

    def calculate_motion_features(self, landmarks_list: List, fps: float) -> List[np.ndarray]:
        """
        计算运动特征（速度、加速度等）

        Args:
            landmarks_list: 关键点列表
            fps: 视频帧率

        Returns:
            运动特征列表
        """
        motion_features = []

        for i in range(len(landmarks_list)):
            if i == 0:
                # 第一帧，速度为0
                motion_features.append(np.zeros(66))  # 33个关键点 * 2 (x, y)
            else:
                current_landmarks = landmarks_list[i]
                prev_landmarks = landmarks_list[i - 1]

                if current_landmarks and prev_landmarks:
                    # 计算速度
                    speeds = []
                    for j in range(min(len(current_landmarks), len(prev_landmarks))):
                        dx = current_landmarks[j].x - prev_landmarks[j].x
                        dy = current_landmarks[j].y - prev_landmarks[j].y
                        # 转换为每秒移动距离
                        speed_x = dx * fps
                        speed_y = dy * fps
                        speeds.extend([speed_x, speed_y])

                    motion_features.append(np.array(speeds))
                else:
                    # 如果关键点缺失，速度为0
                    motion_features.append(np.zeros(66))

        return motion_features

    def detect_action_boundaries(self, landmarks_list: List) -> List[Tuple[int, int]]:
        """
        检测动作起始和结束边界

        Args:
            landmarks_list: 关键点列表

        Returns:
            动作边界列表，每个元素为(start_frame, end_frame)元组
        """
        logger.info("开始检测动作边界...")

        # 计算每帧的特征向量（使用手部关键点的位置变化）
        feature_changes = []

        for i in range(1, len(landmarks_list)):
            prev_landmarks = landmarks_list[i - 1]
            curr_landmarks = landmarks_list[i]

            if prev_landmarks and curr_landmarks:
                # 计算手部关键点的平均位置变化
                total_change = 0.0
                valid_points = 0

                for idx in self.hand_indices:
                    if idx < len(prev_landmarks) and idx < len(curr_landmarks):
                        dx = curr_landmarks[idx].x - prev_landmarks[idx].x
                        dy = curr_landmarks[idx].y - prev_landmarks[idx].y
                        change = np.sqrt(dx ** 2 + dy ** 2)
                        total_change += change
                        valid_points += 1

                if valid_points > 0:
                    avg_change = total_change / valid_points
                else:
                    avg_change = 0.0
            else:
                avg_change = 0.0

            feature_changes.append(avg_change)

        # 使用滑动窗口和平滑处理来识别动作边界
        boundaries = []
        in_action = False
        action_start = 0

        # 简单的阈值检测
        for i, change in enumerate(feature_changes):
            if change > self.segmentation_threshold and not in_action:
                # 动作开始
                action_start = i
                in_action = True
            elif change < self.segmentation_threshold * 0.5 and in_action:
                # 动作结束
                if (i - action_start) >= self.min_action_frames:
                    boundaries.append((action_start, i))
                in_action = False

        # 如果最后一个动作没有结束标记
        if in_action and (len(feature_changes) - action_start) >= self.min_action_frames:
            boundaries.append((action_start, len(feature_changes) - 1))

        logger.info(f"检测到 {len(boundaries)} 个动作")
        return boundaries

    def extract_action_features(self, landmarks_list: List, boundaries: List[Tuple[int, int]], fps: float) -> List[
        Dict[str, Any]]:
        """
        为每个动作片段提取特征向量

        Args:
            landmarks_list: 关键点列表
            boundaries: 动作边界列表
            fps: 视频帧率

        Returns:
            特征向量列表，每个元素包含一个动作的特征
        """
        logger.info("开始提取动作特征...")

        features_list = []

        for start_frame, end_frame in boundaries:
            # 获取动作片段的关键点
            action_landmarks = landmarks_list[start_frame:end_frame + 1]

            # 过滤掉None值
            valid_landmarks = [lm for lm in action_landmarks if lm is not None]
            if len(valid_landmarks) == 0:
                continue

            # 提取多种特征
            # 1. 关键点坐标特征
            coord_features = []
            for landmark in valid_landmarks:
                for idx in self.hand_indices:
                    if idx < len(landmark):
                        coord_features.extend([landmark[idx].x, landmark[idx].y])

            # 2. 关节角度特征
            angle_features = []
            for landmark in valid_landmarks:
                angles = self.calculate_joint_angles(landmark)
                angle_features.extend(angles)

            # 3. 运动速度特征
            motion_features = self.calculate_motion_features(action_landmarks, fps)
            motion_features_flat = []
            for motion_feat in motion_features:
                motion_features_flat.extend(motion_feat)

            # 4. 统计特征（均值、标准差等）
            if len(coord_features) > 0:
                coord_array = np.array(coord_features).reshape(-1, len(self.hand_indices) * 2)
                coord_mean = np.mean(coord_array, axis=0)
                coord_std = np.std(coord_array, axis=0)
                coord_min = np.min(coord_array, axis=0)
                coord_max = np.max(coord_array, axis=0)
            else:
                coord_mean = coord_std = coord_min = coord_max = np.array([])

            if len(angle_features) > 0:
                angle_array = np.array(angle_features).reshape(-1, len(self.hand_joints))
                angle_mean = np.mean(angle_array, axis=0)
                angle_std = np.std(angle_array, axis=0)
                angle_min = np.min(angle_array, axis=0)
                angle_max = np.max(angle_array, axis=0)
            else:
                angle_mean = angle_std = angle_min = angle_max = np.array([])

            # 合并所有特征
            all_features = {
                'coordinates_mean': coord_mean.tolist() if len(coord_mean) > 0 else [],
                'coordinates_std': coord_std.tolist() if len(coord_std) > 0 else [],
                'coordinates_min': coord_min.tolist() if len(coord_min) > 0 else [],
                'coordinates_max': coord_max.tolist() if len(coord_max) > 0 else [],
                'angles_mean': angle_mean.tolist() if len(angle_mean) > 0 else [],
                'angles_std': angle_std.tolist() if len(angle_std) > 0 else [],
                'angles_min': angle_min.tolist() if len(angle_min) > 0 else [],
                'angles_max': angle_max.tolist() if len(angle_max) > 0 else [],
                'motion_features': motion_features_flat,
                'action_start_frame': start_frame,
                'action_end_frame': end_frame,
                'action_duration': (end_frame - start_frame) / fps
            }

            features_list.append(all_features)

        logger.info(f"提取了 {len(features_list)} 个动作的特征")
        return features_list

    def generate_annotated_video(self, input_video: str, output_video: str, boundaries: List[Tuple[int, int]]):
        """
        生成带标注的视频

        Args:
            input_video: 输入视频路径
            output_video: 输出视频路径
            boundaries: 动作边界列表
        """
        logger.info(f"开始生成标注视频: {output_video}")

        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {input_video}")

        # 获取视频参数
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

        current_action_idx = 0
        current_boundary = boundaries[current_action_idx] if boundaries else (0, 0)

        frame_idx = 0
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            # 处理图像并提取关键点
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = self.pose.process(image_rgb)
            image_rgb.flags.writeable = True
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            # 检查当前帧是否在动作边界内
            in_action = (current_boundary[0] <= frame_idx <= current_boundary[1]) if current_boundary else False

            # 绘制姿态关键点（仅在动作期间绘制）
            if results.pose_landmarks and in_action:
                self.mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )

            # 添加动作标注文本
            if in_action:
                action_text = f"Action {current_action_idx + 1}"
                cv2.putText(
                    image,
                    action_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

            # 检查是否需要切换到下一个动作
            if frame_idx == current_boundary[1] and current_action_idx < len(boundaries) - 1:
                current_action_idx += 1
                current_boundary = boundaries[current_action_idx]

            # 写入输出视频
            out.write(image)

            frame_idx += 1
            if frame_idx % 100 == 0:
                logger.info(f"已处理标注视频 {frame_idx}/{total_frames} 帧")

        cap.release()
        out.release()
        logger.info(f"标注视频生成完成: {output_video}")

    def process_video(self, input_video: str, output_video: str, features_output: str):
        """
        完整的视频处理流程

        Args:
            input_video: 输入视频路径
            output_video: 输出标注视频路径
            features_output: 特征向量输出文件路径
        """
        logger.info("开始处理视频...")

        try:
            # 1. 提取关键点
            landmarks_list = self.extract_landmarks(input_video)

            # 2. 获取视频FPS
            cap = cv2.VideoCapture(input_video)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            # 3. 检测动作边界
            boundaries = self.detect_action_boundaries(landmarks_list)

            if not boundaries:
                logger.warning("未检测到任何动作")
                return

            # 4. 提取动作特征
            features_list = self.extract_action_features(landmarks_list, boundaries, fps)

            # 5. 生成标注视频
            self.generate_annotated_video(input_video, output_video, boundaries)

            # 6. 保存特征向量
            with open(features_output, 'w', encoding='utf-8') as f:
                json.dump(features_list, f, ensure_ascii=False, indent=2)

            logger.info(f"处理完成！")
            logger.info(f"- 检测到 {len(boundaries)} 个动作")
            logger.info(f"- 提取了 {len(features_list)} 个动作的特征")
            logger.info(f"- 标注视频保存至: {output_video}")
            logger.info(f"- 特征向量保存至: {features_output}")

        except Exception as e:
            logger.error(f"处理视频时发生错误: {e}")
            raise


def main():
    """
    主函数
    """
    # 配置参数
    input_video_path = "input_video.mp4"  # 输入视频路径
    output_video_path = "annotated_video.mp4"  # 输出标注视频路径
    features_output_path = "action_features.json"  # 特征向量输出路径

    # 检查输入文件是否存在
    if not os.path.exists(input_video_path):
        logger.error(f"输入视频文件不存在: {input_video_path}")
        return

    # 创建处理实例
    segmenter = SignLanguageActionSegmenter(
        detection_confidence=0.5,
        tracking_confidence=0.5
    )

    try:
        # 开始处理
        segmenter.process_video(
            input_video=input_video_path,
            output_video=output_video_path,
            features_output=features_output_path
        )

    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        raise


if __name__ == "__main__":
    main()