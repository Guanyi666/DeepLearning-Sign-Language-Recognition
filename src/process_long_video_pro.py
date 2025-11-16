import cv2
import mediapipe as mp
import numpy as np
import os
import time
from typing import List, Tuple, Optional
import math


class HandSignActionDetector:
    def __init__(self, video_path: str, output_dir: str):
        """
        初始化手语动作检测器

        Args:
            video_path: 输入视频文件路径
            output_dir: 输出目录路径
        """
        # 初始化MediaPipe手部检测器
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        # 注意：在实际使用中，应使用with语句来管理MediaPipe资源
        # 这里我们创建一个实例变量来存储hands对象
        self.hands = None

        # 视频处理参数
        self.video_path = video_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 动作识别参数
        self.motion_threshold = 0.05  # 动作变化阈值
        self.min_action_duration = 0.5  # 最小动作持续时间（秒）
        self.min_frames_per_action = 15  # 最小帧数

        # 内部状态变量
        self.cap = None
        self.fps = 0
        self.frame_count = 0
        self.previous_landmarks = None
        self.current_action_start_time = 0
        self.current_action_start_frame = 0
        self.is_in_action = False
        self.action_segments = []
        self.frame_timestamps = []

        # 用于计算手部运动的参数
        self.motion_buffer = []
        self.motion_buffer_size = 5

    def calculate_hand_motion(self, landmarks1, landmarks2) -> float:
        """
        计算两个手部关键点集合之间的运动距离

        Args:
            landmarks1: 第一个手部关键点集合
            landmarks2: 第二个手部关键点集合

        Returns:
            运动距离值
        """
        if landmarks1 is None or landmarks2 is None:
            return 0.0

        total_distance = 0.0
        landmark_count = 0

        for i in range(len(landmarks1)):
            lm1 = landmarks1[i]
            lm2 = landmarks2[i]

            # 计算关键点之间的欧几里得距离
            distance = math.sqrt(
                (lm1[0] - lm2[0]) ** 2 +
                (lm1[1] - lm2[1]) ** 2 +
                (lm1[2] - lm2[2]) ** 2
            )
            total_distance += distance
            landmark_count += 1

        if landmark_count > 0:
            return total_distance / landmark_count
        else:
            return 0.0

    def is_hand_present(self, landmarks_list) -> bool:
        """
        检查当前帧是否包含手部

        Args:
            landmarks_list: 手部关键点列表

        Returns:
            是否包含手部
        """
        return landmarks_list is not None and len(landmarks_list) > 0

    def detect_action_boundary(self, current_landmarks) -> bool:
        """
        检测动作边界

        Args:
            current_landmarks: 当前帧的手部关键点

        Returns:
            是否检测到动作边界
        """
        if self.previous_landmarks is None:
            return False

        # 计算当前帧与前一帧的运动距离
        motion_distance = self.calculate_hand_motion(current_landmarks, self.previous_landmarks)

        # 检查运动是否低于阈值（表示静止）
        is_stationary = motion_distance < self.motion_threshold

        return is_stationary

    def extract_frame(self, frame, frame_number: int) -> Optional[List]:
        """
        从单帧中提取手部关键点

        Args:
            frame: 视频帧
            frame_number: 帧编号

        Returns:
            手部关键点列表，如果未检测到手则返回None
        """
        # 将BGR图像转换为RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 在这里创建MediaPipe Hands实例以避免全局状态问题
        with self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
        ) as hands:
            # 处理手部关键点
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                # 提取第一个检测到的手部的关键点
                hand_landmarks = results.multi_hand_landmarks[0]
                # 将关键点转换为列表格式
                landmarks_list = []
                for landmark in hand_landmarks.landmark:
                    landmarks_list.append([landmark.x, landmark.y, landmark.z])
                return landmarks_list
            else:
                return None

    def save_action_segment(self, start_frame: int, end_frame: int, action_id: int):
        """
        保存动作片段

        Args:
            start_frame: 起始帧
            end_frame: 结束帧
            action_id: 动作ID
        """
        # 计算时间戳
        start_time = start_frame / self.fps
        end_time = end_frame / self.fps
        duration = end_time - start_time

        # 输出动作信息
        print(f"动作 {action_id}: 帧 {start_frame}-{end_frame}, "
              f"时间 {start_time:.2f}-{end_time:.2f}s, "
              f"持续时间 {duration:.2f}s")

        # 创建输出目录
        action_dir = os.path.join(self.output_dir, f"action_{action_id:03d}")
        os.makedirs(action_dir, exist_ok=True)

        # 重新打开视频以提取帧
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_idx = start_frame
        while frame_idx <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            # 保存帧为图像
            frame_path = os.path.join(action_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(frame_path, frame)

            frame_idx += 1

        cap.release()

        # 保存动作信息到文本文件
        info_path = os.path.join(action_dir, "info.txt")
        with open(info_path, 'w') as f:
            f.write(f"Action ID: {action_id}\n")
            f.write(f"Start Frame: {start_frame}\n")
            f.write(f"End Frame: {end_frame}\n")
            f.write(f"Start Time: {start_time:.2f}s\n")
            f.write(f"End Time: {end_time:.2f}s\n")
            f.write(f"Duration: {duration:.2f}s\n")

    def process_video(self):
        """
        处理整个视频文件
        """
        # 打开视频文件
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print(f"错误：无法打开视频文件 {self.video_path}")
            return

        # 获取视频参数
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"视频参数：FPS={self.fps}, 总帧数={total_frames}")

        # 处理每一帧
        frame_number = 0
        action_id = 1

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # 提取当前帧的手部关键点
            current_landmarks = self.extract_frame(frame, frame_number)

            # 检测动作边界
            is_boundary = self.detect_action_boundary(current_landmarks) if self.previous_landmarks else False

            # 状态机逻辑：检测动作开始和结束
            if current_landmarks is not None and not self.is_in_action:
                # 开始新的动作
                self.current_action_start_frame = frame_number
                self.current_action_start_time = frame_number / self.fps
                self.is_in_action = True
                print(f"检测到动作开始于帧 {frame_number}")

            elif current_landmarks is None and self.is_in_action:
                # 结束当前动作（手部消失）
                action_duration = (frame_number - self.current_action_start_frame) / self.fps
                if action_duration >= self.min_action_duration and \
                        (frame_number - self.current_action_start_frame) >= self.min_frames_per_action:
                    self.save_action_segment(self.current_action_start_frame, frame_number - 1, action_id)
                    action_id += 1
                self.is_in_action = False

            elif is_boundary and self.is_in_action:
                # 检测到静止边界，可能结束当前动作
                action_duration = (frame_number - self.current_action_start_frame) / self.fps
                if action_duration >= self.min_action_duration and \
                        (frame_number - self.current_action_start_frame) >= self.min_frames_per_action:
                    self.save_action_segment(self.current_action_start_frame, frame_number - 1, action_id)
                    action_id += 1
                self.is_in_action = False

            # 更新前一帧关键点
            self.previous_landmarks = current_landmarks

            frame_number += 1

            # 显示进度
            if frame_number % 100 == 0:
                print(f"已处理 {frame_number}/{total_frames} 帧")

        # 处理最后一个动作（如果视频结束时仍在动作中）
        if self.is_in_action:
            action_duration = (frame_number - 1 - self.current_action_start_frame) / self.fps
            if action_duration >= self.min_action_duration and \
                    (frame_number - 1 - self.current_action_start_frame) >= self.min_frames_per_action:
                self.save_action_segment(self.current_action_start_frame, frame_number - 1, action_id)

        # 释放资源
        self.cap.release()

        print(f"处理完成！共识别出 {action_id - 1} 个手语动作。")


def main():
    # 输入输出路径
    video_path = "../data/raw/video/long_hand_video_06.mp4"  # 替换为实际视频路径
    output_dir = "./data/processed/video/segments_06"  # 输出目录

    # 创建检测器实例
    detector = HandSignActionDetector(video_path, output_dir)

    # 开始处理视频
    print("开始处理手语视频...")
    start_time = time.time()

    detector.process_video()

    end_time = time.time()
    print(f"处理完成，总耗时: {end_time - start_time:.2f} 秒")


if __name__ == "__main__":
    main()



