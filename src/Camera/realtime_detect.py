import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os

# ================= 配置区域 =================

# 1. 智能计算模型路径
# 获取当前脚本(realtime_detect.py) 所在的目录: src/Camera
current_dir = os.path.dirname(os.path.abspath(__file__))
# 计算模型的绝对路径: src/Camera -> 上一级(src) -> Conv1D -> h5文件
MODEL_PATH = os.path.join(current_dir, '..', 'Conv1D', 'sign_language_model_v3_tuned.h5')

# 打印一下路径看看对不对（调试用）
print(f"正在尝试加载模型，路径为: {MODEL_PATH}")

# 加载模型
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("模型加载成功！")
except OSError:
    print("❌ 错误：找不到模型文件！请检查路径。")
    exit()
# 2. 定义标签 (非常重要！！！)
# 请查看你 run.py 运行结束时的输出日志 "共 X 个类别: [...]"
# 必须按顺序完全一致地填入这里
actions = ['good', 'he', 'hello', 'morning', 'thank_you', 'you']
# ^^^ 上面这个列表请根据你的实际训练结果修改 ^^^

# 3. 参数设置
SEQUENCE_LENGTH = 30  # 必须与训练时的 seq_length 一致
THRESHOLD = 0.8  # 置信度阈值 (0~1)，越高越严谨

# ===========================================

# 初始化 MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# 核心函数：从检测结果中提取 126 维特征 (与训练预处理逻辑保持一致)
def extract_keypoints(results):
    feature = np.zeros(126)

    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # 获取左右手标签
            # 注意：MediaPipe对于前置摄像头通常是镜像的
            label = results.multi_handedness[idx].classification[0].label

            # 提取 21*3 = 63 个坐标
            # 这里不做归一化，因为你之前的代码也没做额外的数学归一化，只是存了原始坐标
            kp = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

            # 强制规定位置：左手填前63，右手填后63
            if label == 'Left':
                feature[0:63] = kp
            else:
                feature[63:126] = kp

    return feature


# 变量初始化
sequence = []  # 滑动窗口缓冲区
sentence = []  # 屏幕上显示的句子
last_prediction = ""

# 打开摄像头
cap = cv2.VideoCapture(0)

# 设置 MediaPipe (流模式)
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. 图像处理
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 2. 绘制手部骨架
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # 3. 特征提取与预测逻辑
        if results.multi_hand_landmarks:
            # 提取当前帧特征
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)

            # 保持缓冲区长度为 30
            sequence = sequence[-SEQUENCE_LENGTH:]

            # 只有攒够了 30 帧才开始预测
            if len(sequence) == SEQUENCE_LENGTH:
                # 调整维度 [30, 126] -> [1, 30, 126]
                res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]

                # 获取最大概率的类别
                predicted_id = np.argmax(res)
                confidence = res[predicted_id]

                current_action = actions[predicted_id]

                # 简单的可视化逻辑：只显示高置信度的结果
                if confidence > THRESHOLD:
                    # 防抖逻辑：如果动作改变了，才更新显示
                    if current_action != last_prediction:
                        sentence.append(current_action)
                        last_prediction = current_action

                        # 只保留最后 5 个词
                        if len(sentence) > 5:
                            sentence = sentence[-5:]

                # 可以在控制台打印实时概率，方便调试
                # print(f"{current_action}: {confidence:.2f}")

        # 4. 界面 UI 显示
        # 画一个矩形条
        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)

        # 显示识别文字
        cv2.putText(image, ' '.join(sentence), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # 显示当前状态提示
        status_text = "Recording..." if results.multi_hand_landmarks else "No Hand"
        cv2.putText(image, status_text, (10, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        cv2.imshow('Sign Language Recognition Live', image)

        # 按 'q' 退出
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()