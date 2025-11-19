import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from pathlib import Path
import os

# 绘图与评估工具
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# 核心调参工具
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --- 全局环境设置 ---
# 设置中文字体以正确显示绘图标签
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 配置与数据索引 (Configuration & Indexing)
# ==========================================

# 数据集路径
DATA_DIR = Path(r"../../data/dataset/train_aug")

# 留一验证策略 (Leave-One-Subject-Out Validation)
# 目的：评估模型的泛化能力。模型在从未见过的 '04' 号采集者数据上进行测试，
# 防止模型死记硬背特定人的动作习惯。
VALIDATION_SUBJECTS = ['04']

# 超参数配置 (Hyperparameters)
# BATCH_SIZE: 64。权衡内存占用与梯度下降的随机性。
# SEQ_LENGTH: 30。输入的时间步长，对应约1秒的视频数据。
# INPUT_FEATURES: 126。每帧特征维度 (21点 * 2只手 * 3维坐标)。
BATCH_SIZE = 64
SEQ_LENGTH = 30
INPUT_FEATURES = 126

print(f"--- 系统初始化 ---")
print(f"数据源: {DATA_DIR}")
print(f"验证集策略: 使用 {VALIDATION_SUBJECTS} 作为验证集，其余作为训练集。")

train_paths = []
train_labels = []
val_paths = []
val_labels = []

class_to_idx = {}
idx_to_class = {}
current_idx = 0
stats = {}

if not DATA_DIR.exists():
    raise FileNotFoundError(f"错误：找不到数据目录 {DATA_DIR}")

# --- 数据扫描与元数据构建 ---
for class_dir in sorted(DATA_DIR.iterdir()):
    if class_dir.is_dir():
        cname = class_dir.name
        if cname not in class_to_idx:
            class_to_idx[cname] = current_idx
            idx_to_class[current_idx] = cname
            current_idx += 1
            stats[cname] = {'train': 0, 'val': 0}

        cidx = class_to_idx[cname]
        for subject_dir in class_dir.iterdir():
            if subject_dir.is_dir():
                # 判定当前文件夹属于训练集还是验证集
                is_val = subject_dir.name in VALIDATION_SUBJECTS
                for f in subject_dir.glob("*.npy"):
                    if is_val:
                        val_paths.append(str(f))
                        val_labels.append(cidx)
                        stats[cname]['val'] += 1
                    else:
                        train_paths.append(str(f))
                        train_labels.append(cidx)
                        stats[cname]['train'] += 1

print(f"\n数据集统计:")
print(f"训练样本数: {len(train_paths)}")
print(f"验证样本数: {len(val_paths)}")
NUM_CLASSES = len(class_to_idx)
class_names = [idx_to_class[i] for i in range(NUM_CLASSES)]

# --- 类别不平衡处理 (Class Imbalance Handling) ---
# 计算类别权重，赋予稀有类别更高的Loss惩罚，防止模型偏向多数类。
train_labels_np = np.array(train_labels)
val_labels_np = np.array(val_labels)
unique_labels = np.unique(train_labels_np)
class_weights = compute_class_weight('balanced', classes=unique_labels, y=train_labels_np)
class_weight_dict = dict(zip(unique_labels, class_weights))
print(f"类别权重计算完成: {class_weight_dict}")


# ==========================================
# 2. 特征工程管道 (Feature Engineering Pipeline)
# ==========================================

def extract_key_frame(data):
    """
    关键帧提取算法 (Key Frame Extraction Strategy)

    原理：
        通过计算帧间的欧氏距离（速度），寻找动作最“稳定”的时刻。
        通常在手语动作的定格点（Hold Phase），手势形状最清晰，
        且受运动模糊影响最小。

    Args:
        data: Tensor (30, 126) - 原始序列数据

    Returns:
        key_frame: Tensor (126,) - 提取出的静态特征向量
    """
    # 1. 计算差分 (即速度 Velocity): v[t] = x[t+1] - x[t]
    diff = data[1:] - data[:-1]

    # 2. 计算 L2 范数: 将多维特征的变化合并为一个标量速度
    velocity = tf.norm(diff, axis=1)

    # 3. 启发式搜索窗口 (Heuristic Search Window)
    # 忽略动作的起始（起手）和结束（放下）阶段，专注于中间 50% 的核心动作区
    start = SEQ_LENGTH // 4
    end = SEQ_LENGTH * 3 // 4

    # 4. 寻找局部极小值 (Argmin)
    min_idx_local = tf.argmin(velocity[start:end])
    min_idx_global = min_idx_local + start

    key_frame = data[min_idx_global]
    return key_frame


def load_dual_input(path, label):
    """
    TF.Data 数据加载器
    负责将文件路径转换为模型所需的双输入格式。
    """

    def _loader(p):
        # 物理加载 .npy 文件
        d = np.load(p.numpy().decode('utf-8')).astype(np.float32)
        return d

    # 1. 加载原始时序数据
    seq_data = tf.py_function(_loader, [path], tf.float32)
    seq_data.set_shape((SEQ_LENGTH, INPUT_FEATURES))

    # 2. 在线计算静态关键帧 (On-the-fly processing)
    static_data = extract_key_frame(seq_data)
    static_data.set_shape((INPUT_FEATURES,))

    label = tf.cast(label, tf.int64)

    # 3. 构造双输入字典 (Dual-Input Dictionary)
    # 键名 'seq_input' 和 'static_input' 必须与模型 Input 层的 name 一致
    return {'seq_input': seq_data, 'static_input': static_data}, label


# --- 构建高性能数据管道 (Data Pipeline Construction) ---
# AUTOTUNE: 让 TensorFlow 自动根据 CPU 核心数调整并行程度
train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_ds = train_ds.shuffle(len(train_paths)) \
    .map(load_dual_input, num_parallel_calls=tf.data.AUTOTUNE) \
    .batch(BATCH_SIZE) \
    .prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_ds = val_ds.map(load_dual_input, num_parallel_calls=tf.data.AUTOTUNE) \
    .batch(BATCH_SIZE) \
    .prefetch(tf.data.AUTOTUNE)


# ==========================================
# 3. 双流网络架构搭建 (Model Architecture)
# ==========================================

def create_dual_stream_model(input_shape_seq, input_shape_static, num_classes):
    """
    搭建 Two-Stream CNN-LSTM 融合网络
    """
    # ---------------------------------------------------------
    # Branch A: 动态流 (Dynamic Stream)
    # 目的：提取时空特征 (Spatio-Temporal Features)
    # ---------------------------------------------------------
    input_seq = layers.Input(shape=input_shape_seq, name='seq_input')

    # Feature Extraction (CNN): 提取局部短时特征
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(input_seq)
    x = layers.BatchNormalization()(x)  # 抑制梯度消失，加速收敛
    x = layers.MaxPooling1D(2)(x)  # 降维，保留显著特征

    x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # Sequence Modeling (Bi-LSTM): 提取长时依赖
    # Bidirectional 允许模型同时利用过去和未来的信息
    # return_sequences=False: 只输出序列最后的状态，作为整个序列的摘要
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=False))(x)
    x = layers.Dropout(0.4)(x)  # 正则化，防止过拟合

    dynamic_features = layers.Dense(64, activation='relu')(x)

    # ---------------------------------------------------------
    # Branch B: 静态流 (Static Stream)
    # 目的：提取纯空间手型特征 (Spatial Pose Features)
    # ---------------------------------------------------------
    input_static = layers.Input(shape=input_shape_static, name='static_input')

    # Pose Encoding (MLP): 将坐标映射到高维语义空间
    y = layers.Dense(128, activation='relu')(input_static)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.3)(y)

    y = layers.Dense(64, activation='relu')(y)
    y = layers.BatchNormalization()(y)

    static_features = layers.Dense(64, activation='relu')(y)

    # ---------------------------------------------------------
    # Fusion & Classification (融合与分类)
    # ---------------------------------------------------------
    # Feature Fusion: 拼接动态和静态特征向量
    combined = layers.concatenate([dynamic_features, static_features])

    # Classification Head
    z = layers.Dense(128, activation='relu')(combined)
    z = layers.Dropout(0.5)(z)  # High Dropout due to small dataset size
    output = layers.Dense(num_classes, activation='softmax')(z)

    model = keras.Model(inputs=[input_seq, input_static], outputs=output, name="TwoStream_SignNet")
    return model


# 实例化并打印模型概况
model = create_dual_stream_model((SEQ_LENGTH, INPUT_FEATURES), (INPUT_FEATURES,), NUM_CLASSES)
model.summary()

# ==========================================
# 4. 训练与回调策略 (Training & Callbacks)
# ==========================================

EPOCHS = 500
# 学习率设置 (0.0001): 较小的学习率有助于在复杂的多模态损失面上找到更优的极小值，
# 虽然收敛较慢，但能减少震荡。
LEARNING_RATE = 0.0001

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_list = [
    # 早停机制: 防止过拟合。如果验证集 Loss 在 40 轮内没有下降，则停止。
    # restore_best_weights=True 确保模型回滚到性能最好的状态，而不是最后的状态。
    EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True, verbose=1),

    # 模型检查点: 始终保存验证集 Loss 最低的模型版本。
    ModelCheckpoint("./sign_language_model_dual.h5", monitor='val_loss', save_best_only=True, verbose=0)
]

print("\n--- 开始训练 (Training Phase) ---")
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    class_weight=class_weight_dict,  # 应用类别权重，解决样本不平衡
    callbacks=callbacks_list
)


# ==========================================
# 5. 评估与可视化 (Evaluation & Visualization)
# ==========================================

def plot_history(history):
    """绘制训练曲线，用于诊断 过拟合/欠拟合"""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    best_epoch = np.argmin(val_loss)

    plt.figure(figsize=(15, 6))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Train Acc')
    plt.plot(val_acc, label='Val Acc', linestyle='--')
    plt.scatter(best_epoch, val_acc[best_epoch], c='red', label='Best Epoch')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss', linestyle='--')
    plt.scatter(best_epoch, val_loss[best_epoch], c='red', label='Best Epoch')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_history_dual.png")
    plt.show()


plot_history(history)

print("\n--- 最终测试报告 (Final Evaluation on Validation Set) ---")
# 预测
y_pred_probs = model.predict(val_ds)
y_pred = np.argmax(y_pred_probs, axis=1)

# 分类报告
print(classification_report(val_labels_np, y_pred, target_names=class_names, zero_division=0))

# 混淆矩阵可视化
plt.figure(figsize=(10, 8))
cm = confusion_matrix(val_labels_np, y_pred)
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Dual Stream)')
plt.show()