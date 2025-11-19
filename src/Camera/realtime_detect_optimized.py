import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os
import time
from collections import Counter

# ================= 配置区域 =================

# 1. 路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(current_dir, '..', 'Conv1D', 'sign_language_model_dual.h5')

# 2. 动作标签 (请确保顺序与 run.py 输出完全一致)
actions = ['good', 'he', 'morning', 'thank_you', 'very', 'you']

# 3. 核心参数调整
SEQUENCE_LENGTH = 30  # 必须与训练一致
CONFIDENCE_THRESHOLD = 0.6  # [优化] 稍微调低一点，提高灵敏度
STABILITY_FRAMES = 10  # [新增] 连续多少帧预测一致才算确认？(防抖)
ACTION_COOLDOWN = 1.5  # [新增] 动作确认后的冷却时间(秒)，防止重复触发

# ===========================================

# 加载模型
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ 模型加载成功")
except:
    print("❌ 找不到模型文件")
    exit()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def extract_keypoints(results):
    """特征提取逻辑 (必须保持一致)"""
    feature = np.zeros(126)
    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            label = results.multi_handedness[idx].classification[0].label
            kp = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            if label == 'Left':
                feature[0:63] = kp
            else:
                feature[63:126] = kp
    return feature


def extract_key_frame_numpy(data):
    """
    [新增] 从30帧数据中提取最静止的一帧 (关键帧)
    完全复刻 run2.py 的逻辑，但使用 numpy 以提升推理速度
    """
    data = np.array(data)  # 确保是 numpy 数组
    seq_len = len(data)

    # 1. 计算相邻帧的变化速度 (欧氏距离)
    # data[1:] - data[:-1] 得到相邻帧的差值
    diff = data[1:] - data[:-1]
    # 求 L2 范数得到速度
    velocity = np.linalg.norm(diff, axis=1)

    # 2. 限制寻找范围 (中间 50%)
    start = seq_len // 4
    end = seq_len * 3 // 4

    # 3. 找到速度最小的那一帧 (动作最稳的时候)
    # velocity[start:end] 是截取出的速度片段
    min_idx_local = np.argmin(velocity[start:end])
    min_idx_global = min_idx_local + start

    return data[min_idx_global]



# 状态变量
sequence = []  # 存储最近30帧特征
predictions_queue = []  # 存储最近几帧的预测结果(用于投票)
sentence = []  # 最终显示的句子
last_action_time = 0  # 上次动作触发的时间戳

cap = cv2.VideoCapture(0)

# 开启 MediaPipe
with mp_hands.Hands(
        model_complexity=0,  # [优化] 0=Lite模型(最快), 1=Full
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 镜像翻转 (让自己看像照镜子，但注意MediaPipe的左右手逻辑)
        # frame = cv2.flip(frame, 1)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 绘制
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # ================== 核心识别逻辑 ==================
        if results.multi_hand_landmarks:
            # 1. 提取数据
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-SEQUENCE_LENGTH:]  # 保持30帧

            # 2. 只有攒够数据才预测
            # 只有攒够数据才预测
            if len(sequence) == SEQUENCE_LENGTH:
                # --- 数据准备 ---
                # 1. 动态流输入 (Sequence): (1, 30, 126)
                input_seq = np.expand_dims(sequence, axis=0)

                # 2. 静态流输入 (Key Frame): (1, 126)
                # 调用刚才写好的函数，从当前30帧里算出最稳的那一帧
                key_frame = extract_key_frame_numpy(sequence)
                input_static = np.expand_dims(key_frame, axis=0)

                # --- 模型预测 ---
                # 注意：这里传入一个列表 [seq, static]，顺序必须和 run2.py 里 model inputs 定义的顺序一致
                res = model([input_seq, input_static], training=False)[0].numpy()

                # 获取结果
                curr_class_id = np.argmax(res)
                curr_conf = res[curr_class_id]

                # --- 下面是原本的防抖/投票逻辑 (保持不变) ---
                if curr_conf > CONFIDENCE_THRESHOLD:
                    predictions_queue.append(curr_class_id)
                else:
                    predictions_queue.append(-1)

                # ... (后续代码不用动)

                predictions_queue = predictions_queue[-STABILITY_FRAMES:]

                # 4. 确认动作 (连句逻辑)
                # 只有当队列里大部分(比如80%)都是同一个结果时，才认为是一个稳态动作
                if len(predictions_queue) == STABILITY_FRAMES:
                    # 统计出现次数最多的结果
                    most_common_id, count = Counter(predictions_queue).most_common(1)[0]

                    # 条件A: 队列里大部分是这个动作
                    # 条件B: 不是 -1 (无效动作)
                    # 条件C: 冷却时间已过 (防止同一个动作连续输出)
                    if (most_common_id != -1) and \
                            (count > STABILITY_FRAMES * 0.8) and \
                            (time.time() - last_action_time > ACTION_COOLDOWN):

                        action_name = actions[most_common_id]

                        # 添加到句子
                        sentence.append(action_name)
                        print(f"识别成功: {action_name}")

                        # 限制句子长度
                        if len(sentence) > 5:
                            sentence = sentence[-5:]

                        # 更新冷却时间
                        last_action_time = time.time()
                        # 清空稳定性队列，防止残影
                        predictions_queue = []

                        # ================== 界面显示 ==================

        # 顶部背景条
        cv2.rectangle(image, (0, 0), (640, 45), (245, 117, 16), -1)

        # 显示句子
        text_display = ' -> '.join(sentence)
        cv2.putText(image, text_display, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 显示状态 (可选：显示冷却状态)
        if time.time() - last_action_time < ACTION_COOLDOWN:
            cv2.putText(image, "Cooldown...", (10, 460),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        else:
            cv2.putText(image, "Ready", (10, 460),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        cv2.imshow('Sign Language AI (Optimized)', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()