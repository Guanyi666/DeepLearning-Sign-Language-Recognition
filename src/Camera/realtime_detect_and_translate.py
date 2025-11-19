import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os
import time
from collections import Counter
from translator import SignTranslator  # 导入刚才写的类
from PIL import Image, ImageDraw, ImageFont
from nlg_processor import NLGProcessor  # 导入刚才新建的模块


# ================= 配置区域 =================

# 1. 路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(current_dir, '..', 'Conv1D', 'sign_language_model_dual (1).h5')

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



def cv2_add_chinese_text(img, text, position, text_color=(0, 255, 0), text_size=30):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    draw = ImageDraw.Draw(img)
    # 必须确保你的电脑里有中文字体文件，例如 simhei.ttf
    # 如果没有，Windows通常在 C:/Windows/Fonts/simhei.ttf
    try:
        fontStyle = ImageFont.truetype("simhei.ttf", text_size, encoding="utf-8")
    except:
        # 如果找不到字体，回退默认（可能不显示中文）
        fontStyle = ImageFont.load_default()

    draw.text(position, text, fill=text_color, font=fontStyle)

    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


# =========================================================
# 外部接口：这里处理最终生成的动作列表
# =========================================================
def process_final_sequence(action_list):
    """
    当用户停止动作 2.5 秒后，这个函数会被自动调用。
    action_list: 例如 ['morning', 'good']
    """
    print(f"\n>>> [外部接口触发] 原始序列: {action_list}")

    # 调用 NLG 处理器进行生成
    # 输入: ['morning', 'good'] -> 输出: "早上好"
    final_sentence = nlg_engine.process(action_list)

    return final_sentence



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

# ...
translator = SignTranslator() # (可选：如果你只用NLG，这个旧翻译器甚至可以删了)
nlg_engine = NLGProcessor()   # <--- 初始化新的 NLG 处理器
# ...
current_sentence = ""         # 用来显示翻译后的中文句子
cap = cv2.VideoCapture(0)

# 开启 MediaPipe
with mp_hands.Hands(
        model_complexity=0,  # [优化] 0=Lite模型(最快), 1=Full
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 图像预处理
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 绘制骨架
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # =========================================================
        # Core: 核心识别逻辑 (只有检测到手时才运行)
        # =========================================================
        if results.multi_hand_landmarks:
            # 1. 提取数据
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-SEQUENCE_LENGTH:]  # 保持30帧

            # 2. 只有攒够数据才预测
            if len(sequence) == SEQUENCE_LENGTH:
                # --- 数据准备 (双流模型) ---
                input_seq = np.expand_dims(sequence, axis=0)  # (1, 30, 126)
                key_frame = extract_key_frame_numpy(sequence)
                input_static = np.expand_dims(key_frame, axis=0)  # (1, 126)

                # --- 模型预测 ---
                res = model([input_seq, input_static], training=False)[0].numpy()

                # 获取结果
                curr_class_id = np.argmax(res)
                curr_conf = res[curr_class_id]

                # --- 防抖/投票逻辑 ---
                if curr_conf > CONFIDENCE_THRESHOLD:
                    predictions_queue.append(curr_class_id)
                else:
                    predictions_queue.append(-1)

                predictions_queue = predictions_queue[-STABILITY_FRAMES:]

                # 3. 确认动作 (连句逻辑)
                if len(predictions_queue) == STABILITY_FRAMES:
                    most_common_id, count = Counter(predictions_queue).most_common(1)[0]

                    # 核心判断：有效动作 + 频率高 + 冷却时间已过
                    if (most_common_id != -1) and \
                            (count > STABILITY_FRAMES * 0.8) and \
                            (time.time() - last_action_time > ACTION_COOLDOWN):

                        action_name = actions[most_common_id]

                        # === 翻译器接入 ===
                        print(f"识别到单词: {action_name}")

                        # 把词喂给翻译器
                        is_new = translator.add_word(action_name)

                        # 如果加了新词，更新屏幕上显示的“单词流”
                        if is_new:
                            sentence = translator.word_buffer

                            # 状态重置
                        last_action_time = time.time()
                        predictions_queue = []  # 清空队列防止连发

        # =========================================================
        # Global Logic: 全局逻辑
        # =========================================================

        # 1. [修改] 检查是否超时提交
        # 这里返回的是 原始列表 (Raw List)，例如 ['you', 'good']
        captured_sequence = translator.check_auto_submit()

        if captured_sequence:
            # -----------------------------------------------------
            # 核心点：列表拿到了！在这里调用你的“其他代码”
            # -----------------------------------------------------
            final_result = process_final_sequence(captured_sequence)

            # 更新 UI 显示
            current_sentence = final_result
            print(f"处理结果: {current_sentence}")

            # 清空屏幕上的英文流 (因为它已经提交处理了)
            sentence = []

        # 2. 界面绘制 (保持不变)
        # A. 背景条
        cv2.rectangle(image, (0, 0), (640, 85), (245, 117, 16), -1)

        # B. 英文流 (实时显示的正在构建的句子)
        text_display = ' + '.join(sentence)
        cv2.putText(image, text_display, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # C. 中文翻译 (显示上一次处理的结果)
        if current_sentence:
            image = cv2_add_chinese_text(image, f"翻译: {current_sentence}", (10, 40), (255, 255, 0), 30)

        # ... (后续的状态显示和 imshow 不变)

        # D. 状态显示
        if time.time() - last_action_time < ACTION_COOLDOWN:
            cv2.putText(image, "Cooldown...", (10, 460),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        else:
            cv2.putText(image, "Ready", (10, 460),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        # 3. 显示
        cv2.imshow('Sign Language AI (Optimized)', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()