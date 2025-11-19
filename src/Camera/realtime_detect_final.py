import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os
import time
from collections import Counter
from threading import Thread

# 引入你的模块
from translator import SignTranslator
from nlg_processor import NLGProcessor


# =========================================================
# 类定义：多线程摄像头读取 (解决画面延迟的核心)
# =========================================================
class WebcamStream:
    def __init__(self, src=0):
        # 开启摄像头
        self.stream = cv2.VideoCapture(src)
        # 设置分辨率 (可选，降低分辨率能显著提速)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()


# =========================================================
# 配置区域
# =========================================================
# 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(current_dir, '..', 'Conv1D', 'sign_language_model_final.h5')

# 标签 (必须与训练一致)
actions = ['good', 'he', 'morning', 'thank_you', 'very', 'you']

# 参数
SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.5  # 稍微降低阈值
STABILITY_FRAMES = 5  # 降低防抖帧数，提升响应速度
ACTION_COOLDOWN = 1.5

# 优化参数
SKIP_FRAMES = 2  # 每隔 2 帧处理一次 MediaPipe (提速)
EMA_ALPHA = 0.6  # 平滑系数 (防抖)

# =========================================================
# 初始化
# =========================================================
# 1. 加载模型
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ 模型加载成功")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    exit()

# 2. 初始化工具
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
translator = SignTranslator()
nlg_engine = NLGProcessor()

# 3. 全局状态变量
sequence = []
predictions_queue = []
sentence = []
last_action_time = 0
no_hand_frames_count = 0  # 记录手消失了多久

# [EMA全局变量] 记录上一帧的平滑坐标
prev_smoothed_kp = np.zeros(126)


# =========================================================
# 辅助函数
# =========================================================
def extract_keypoints_smoothed(results):
    """
    结合了 EMA 平滑的特征提取
    """
    global prev_smoothed_kp  # 引用全局变量，防止重置

    current_kp = np.zeros(126)

    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # 安全获取 label (防止越界)
            if idx < len(results.multi_handedness):
                label = results.multi_handedness[idx].classification[0].label
                kp = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

                if label == 'Left':
                    current_kp[0:63] = kp
                else:
                    current_kp[63:126] = kp

    # 如果检测不到手，current_kp 是全0
    # 如果是全0，我们不应该让它平滑过渡，而是直接归零，否则会“拖尾”
    if np.all(current_kp == 0):
        prev_smoothed_kp = np.zeros(126)
        return np.zeros(126)

    # EMA 公式: S_t = α * Y_t + (1-α) * S_{t-1}
    smoothed_kp = (EMA_ALPHA * current_kp) + ((1 - EMA_ALPHA) * prev_smoothed_kp)
    prev_smoothed_kp = smoothed_kp

    return smoothed_kp


def extract_key_frame_numpy(data):
    """提取关键帧 (双流网络专用)"""
    data = np.array(data)
    if len(data) < 2: return data[0]  # 保护逻辑

    diff = data[1:] - data[:-1]
    velocity = np.linalg.norm(diff, axis=1)

    start = len(data) // 4
    end = len(data) * 3 // 4

    if start >= end: return data[len(data) // 2]  # 保护逻辑

    min_idx_local = np.argmin(velocity[start:end])
    min_idx_global = min_idx_local + start
    return data[min_idx_global]


def cv2_add_chinese_text(img, text, position, text_color=(0, 255, 0), text_size=30):
    from PIL import Image, ImageDraw, ImageFont
    if (isinstance(img, np.ndarray)):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    try:
        fontStyle = ImageFont.truetype("simhei.ttf", text_size, encoding="utf-8")
    except:
        fontStyle = ImageFont.load_default()
    draw.text(position, text, fill=text_color, font=fontStyle)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def process_final_sequence(action_list):
    print(f"\n>>> [外部接口触发] 原始序列: {action_list}")
    final_sentence = nlg_engine.process(action_list)
    return final_sentence


# =========================================================
# 主循环
# =========================================================

# 1. 开启多线程摄像头
cap = WebcamStream(src=0).start()
print(">>> 摄像头已启动 (多线程模式)")
time.sleep(1.0)  # 给摄像头一点预热时间

frame_count = 0
last_results = None  # 缓存上一帧的检测结果

# 开启 MediaPipe
with mp_hands.Hands(
        model_complexity=0,  # 0最快
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while True:
        # 从线程读取最新帧
        frame = cap.read()
        if frame is None: break

        frame_count += 1

        # 图像预处理
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # =================================================
        # 优化 1: 跳帧处理 (Frame Skipping)
        # =================================================
        # 逻辑：如果当前是“处理帧”，跑 MediaPipe；如果不是，直接沿用上一帧的 last_results
        # 这样既省了 CPU，又保证了逻辑连续，不会因为 results 为空导致误判

        if frame_count % SKIP_FRAMES == 0:
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            last_results = results  # 更新缓存
        else:
            # 复用上一帧结果
            results = last_results
            # 如果刚启动还没跑过 MediaPipe，构造一个空的伪对象
            if results is None:
                class EmptyResults: multi_hand_landmarks = None


                results = EmptyResults()

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 绘制 (只在有结果时绘制)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # =================================================
        # 逻辑修复: 正确的缓存重置逻辑
        # =================================================

        # 我们使用 results (可能是复用的) 来判断有没有手
        has_hand = results.multi_hand_landmarks is not None

        if has_hand:
            # 如果检测到了手
            if no_hand_frames_count > 10:
                sequence = []  # 真的离开很久了，才清空
                predictions_queue = []
                print(">>> 检测到新输入，缓存已重置")

            no_hand_frames_count = 0  # 重置计数器
        else:
            # 没检测到手
            no_hand_frames_count += 1
            # 如果长时间没手，清理 sequence，防止下次误判
            if no_hand_frames_count > 20:
                sequence = []

        # =================================================
        # 核心识别 (结合 EMA 和 双流网络)
        # =================================================
        if has_hand:
            # 1. 提取数据 (即使是复用的 results，也要跑这一步，因为 EMA 需要连续计算)
            keypoints = extract_keypoints_smoothed(results)
            sequence.append(keypoints)
            sequence = sequence[-SEQUENCE_LENGTH:]  # 保持30帧

            # 2. 预测
            if len(sequence) == SEQUENCE_LENGTH:
                # 准备双流输入
                input_seq = np.expand_dims(sequence, axis=0)
                key_frame = extract_key_frame_numpy(sequence)
                input_static = np.expand_dims(key_frame, axis=0)

                # 预测
                res = model([input_seq, input_static], training=False)[0].numpy()
                curr_class_id = np.argmax(res)
                curr_conf = res[curr_class_id]

                # 投票
                if curr_conf > CONFIDENCE_THRESHOLD:
                    predictions_queue.append(curr_class_id)
                else:
                    predictions_queue.append(-1)

                predictions_queue = predictions_queue[-STABILITY_FRAMES:]

                # 确认动作
                if len(predictions_queue) == STABILITY_FRAMES:
                    most_common_id, count = Counter(predictions_queue).most_common(1)[0]

                    if (most_common_id != -1) and \
                            (count > STABILITY_FRAMES * 0.8) and \
                            (time.time() - last_action_time > ACTION_COOLDOWN):

                        action_name = actions[most_common_id]
                        print(f"识别: {action_name}")

                        # 翻译处理
                        if translator.add_word(action_name):
                            sentence = translator.word_buffer

                        last_action_time = time.time()
                        predictions_queue = []

        # =================================================
        # 全局逻辑 & UI
        # =================================================

        # 自动翻译结算
        captured_sequence = translator.check_auto_submit()
        if captured_sequence:
            final_result = process_final_sequence(captured_sequence)
            sentence = []  # 清空英文流
            # 这里的 final_result 显示由下面的 UI 逻辑接管，我们存到全局变量里供显示
            # 但为了简单，我们复用 current_sentence 变量
            # 你的 UI 代码里可能叫 current_sentence，这里我们在 translator 里没存
            # 我们直接修改 SignTranslator 让他返回结果时把结果存到 translator 内部或者外部变量
            # 这里简单处理：
            translator.current_translation_display = final_result

        # 绘制 UI
        cv2.rectangle(image, (0, 0), (640, 85), (245, 117, 16), -1)

        # 英文流
        text_display = ' + '.join(sentence)
        cv2.putText(image, text_display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 中文翻译 (假设 translator 对象上挂载一个属性用来显示，或者你用全局变量)
        if hasattr(translator, 'current_translation_display'):
            image = cv2_add_chinese_text(image, f"翻译: {translator.current_translation_display}", (10, 40),
                                         (255, 255, 0), 30)

        # 状态
        if time.time() - last_action_time < ACTION_COOLDOWN:
            cv2.putText(image, "Cooldown", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        else:
            cv2.putText(image, "Ready", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        cv2.imshow('Sign Language AI (Ultimate)', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 结束
cap.stop()
cv2.destroyAllWindows()