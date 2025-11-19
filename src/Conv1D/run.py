import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import time
import os

# 绘图与评估工具
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# 核心调参工具
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping

# --- 全局绘图设置 ---
# 指定中文字体（Windows系统常用“SimHei”或“Microsoft YaHei”）
rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ==========================================
# 1. 数据准备与预处理
# ==========================================

# ！！！确保这个路径指向你正确的 'train' 文件夹！！！
DATA_DIR = Path(r"../../data/dataset/train")

print("开始扫描数据文件...")

all_filepaths = []
all_labels = []
class_to_idx = {}
idx_to_class = {}
current_idx = 0

# 遍历所有类别文件夹
for class_dir in DATA_DIR.iterdir():
    if class_dir.is_dir():
        class_name = class_dir.name
        if class_name not in class_to_idx:
            class_to_idx[class_name] = current_idx
            idx_to_class[current_idx] = class_name
            current_idx += 1

        class_idx = class_to_idx[class_name]
        for subject_dir in class_dir.iterdir():
            if subject_dir.is_dir():
                for npy_file in subject_dir.glob("*.npy"):
                    all_filepaths.append(str(npy_file))
                    all_labels.append(class_idx)

print(f"扫描完成！总共找到 {len(all_filepaths)} 个特征文件。")
NUM_CLASSES = len(class_to_idx)
print(f"共 {NUM_CLASSES} 个类别: {list(class_to_idx.keys())}")
class_names = [idx_to_class[i] for i in range(NUM_CLASSES)]

# --- [调参核心 1] 数据集划分策略 ---
# 这里的 stratify=all_labels 非常重要。
# 如果不加这一项，随机切分可能导致验证集中某些稀有类别的样本为0。
# stratify 保证训练集和验证集中各类别的比例是一致的。
train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_filepaths,
    all_labels,
    test_size=0.2,
    random_state=42,
    stratify=all_labels
)
print(f"训练集样本数: {len(train_paths)}")
print(f"验证集样本数: {len(val_paths)}")

# --- [调参核心 2] 类别权重 (Class Weights) ---
# 目的：解决 "Thank you" 样本多、"Morning" 样本少导致的预测偏见。
# 原理：算出每个类别的“稀缺度”。样本越少，权重越大。
# 效果：模型如果预测错了一个稀缺样本，Loss惩罚会比预测错一个常见样本大得多。
print("\n--- 正在计算类别权重 ---")
unique_labels = np.unique(train_labels)
class_weights = compute_class_weight(
    'balanced',
    classes=unique_labels,
    y=train_labels
)
class_weight_dict = dict(zip(unique_labels, class_weights))
print("类别权重字典 (样本少的类权重高):", class_weight_dict)

# --- 构建 tf.data 数据管道 ---
BATCH_SIZE = 64
SEQ_LENGTH = 30
INPUT_FEATURES = 126
INPUT_SHAPE = (SEQ_LENGTH, INPUT_FEATURES)


def load_npy_file(path, label):
    def _loader(path):
        data = np.load(path.numpy().decode('utf-8')).astype(np.float32)
        return data

    data = tf.py_function(_loader, [path], tf.float32)
    data.set_shape(INPUT_SHAPE)
    label = tf.cast(label, tf.int64)
    return data, label


train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_ds = train_ds.shuffle(buffer_size=len(train_paths))
train_ds = train_ds.map(load_npy_file, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_ds = val_ds.map(load_npy_file, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# ==========================================
# 2. 模型架构设计 (调参重点)
# ==========================================

def create_DEEPER_cnn_lstm_model(input_shape, num_classes):
    """
    [调参核心 3] 深层网络架构设计
    - 为什么用 Conv1D? 用于提取单帧内的手部空间特征（如手指弯曲程度）。
    - 为什么加深层数? 3层卷积能提取从简单到抽象的高级特征。
    - 为什么用 Bidirectional LSTM? 手语是连贯动作，双向LSTM能同时利用“过去”和“未来”的信息。
    - 为什么 Dropout 高达 0.4/0.5? 强力抑制过拟合，强迫模型不依赖单一特征。
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),  # (30, 126)

        # --- 特征提取部分 (Spatial Feature Extraction) ---
        layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),  # 加速收敛，防止梯度消失

        layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),

        layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),

        layers.Conv1D(filters=256, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),

        # --- 时序学习部分 (Temporal Sequence Learning) ---
        layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
        layers.Dropout(0.4),  # 丢弃 40% 神经元

        layers.Bidirectional(layers.LSTM(128)),
        layers.Dropout(0.4),

        # --- 分类头 ---
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),  # 全连接层通常需要更高的 Dropout

        layers.Dense(num_classes, activation='softmax')
    ])
    return model


model = create_DEEPER_cnn_lstm_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
model.summary()

# ==========================================
# 3. 编译与训练配置 (调参重点)
# ==========================================

EPOCHS = 500
# [调参核心 4] 学习率 (Learning Rate)
# 默认是 0.001。我们降到 0.00003 (3e-5)。
# 原因：复杂模型 + 少量数据 = 容易震荡。低学习率让模型“步步为营”，虽然慢，但能找到更好的最优解。
LEARNING_RATE = 0.00003

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# [调参核心 5] 早停机制 (Early Stopping)
# 原因：我们设置了 500 epoch，但模型可能在 100 epoch 就过拟合了（训练集准，验证集降）。
# 策略：如果验证集 Loss 在 50 个 epoch 内都不再下降，就强行停止，并回滚到最好的那一版参数。
early_stopping_cb = EarlyStopping(
    monitor='val_loss',
    patience=50,  # 容忍度：允许 50 次尝试而不进步
    restore_best_weights=True,  # 关键：恢复到最佳模型，而不是最后一次的模型
    verbose=1
)

print("\n--- 开始训练 (已加入 Class Weights 和 EarlyStopping) ---")
start_time = time.time()

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    class_weight=class_weight_dict,  # 应用类别权重
    callbacks=[early_stopping_cb]  # 应用早停
)

end_time = time.time()
print(f"--- 训练完成 --- (总耗时: {end_time - start_time:.2f} 秒)")

# 保存模型
save_path = r"./sign_language_model_v3_tuned.h5"
model.save(save_path)
print(f"模型已保存为 '{save_path}'")


# ==========================================
# 4. [新增模块] 训练过程可视化
# ==========================================

def plot_training_history(history, save_name="training_history.png"):
    """
    绘制训练集与验证集的 Accuracy 和 Loss 曲线。
    这是判断模型状态（过拟合/欠拟合）最直观的图表。
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # 找到验证集Loss最低的那个epoch（最佳点）
    best_epoch = np.argmin(val_loss)
    best_val_loss = val_loss[best_epoch]
    best_val_acc = val_acc[best_epoch]

    epochs_range = range(len(acc))

    plt.figure(figsize=(15, 6))

    # --- 子图 1: 准确率 ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy', linestyle='--')
    # 标记最佳点
    plt.scatter(best_epoch, best_val_acc, c='red', zorder=5)
    plt.annotate(f'Best: {best_val_acc:.2%}', (best_epoch, best_val_acc),
                 xytext=(0, 10), textcoords='offset points', ha='center')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)

    # --- 子图 2: 损失值 ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss', linestyle='--')
    # 标记最佳点
    plt.scatter(best_epoch, best_val_loss, c='red', zorder=5)
    plt.annotate(f'Min Loss: {best_val_loss:.4f}', (best_epoch, best_val_loss),
                 xytext=(0, 10), textcoords='offset points', ha='center')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_name)
    print(f"[可视化] 训练曲线图已保存: {save_name}")
    plt.show()


# 调用可视化函数
print("\n--- 正在生成训练过程分析图 ---")
plot_training_history(history, save_name="training_process_analysis.png")

# ==========================================
# 5. 最终评估 (混淆矩阵与报告)
# ==========================================

print("\n--- 正在生成详细评估报告 ---")
# 获取所有真实标签
y_true = []
for features, labels in val_ds:
    y_true.extend(labels.numpy())
y_true = np.array(y_true)

# 获取所有预测标签
predictions_raw = model.predict(val_ds)
y_pred = np.argmax(predictions_raw, axis=1)

# 打印分类报告
print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

# 绘制混淆矩阵
print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt='g',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.title('Confusion Matrix (Final Tuned Model)', fontsize=15)
plt.show()