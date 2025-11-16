import cv2
import numpy as np
import mediapipe as mp


class SingleHandFeatureExtractor:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        # 用于视频逐帧处理，必须 static_image_mode=True
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_frame(self, frame):
        """从单帧图像提取 126 维特征"""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        feature = np.zeros(126)
        idx = 0

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                for lm in hand.landmark:
                    if idx + 2 < 126:
                        feature[idx] = lm.x
                        feature[idx + 1] = lm.y
                        feature[idx + 2] = lm.z
                        idx += 3
                if idx >= 126:
                    break

        return feature

    def draw_landmarks(self, frame):
        """在图像上绘制关键点"""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for lm in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, lm, self.mp_hands.HAND_CONNECTIONS)

        return frame
