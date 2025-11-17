import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import time
import os

# 导入评估工具
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 导入我们的修复工具
from sklearn.utils.class_weight import compute_class_weight # <--- [FIX 1] 类别权重
from tensorflow.keras.callbacks import EarlyStopping      # <--- [FIX 2] 早停

# --- 第 1 步：扫描数据文件 ---
# ！！！确保这个路径指向你正确的 'train' 文件夹！！！
DATA_DIR = Path(r"D:\code\deeplearning\project\DeepLearning-Sign-Language-Recognition\data\dataset\train")

print("开始扫描数据文件...")

all_filepaths = []
all_labels = []
class_to_idx = {}
idx_to_class = {}
current_idx = 0

# 遍历所有类别文件夹 (good, he, morning, ...)
for class_dir in DATA_DIR.iterdir():
    if class_dir.is_dir():
        class_name = class_dir.name
        if class_name not in class_to_idx:
            class_to_idx[class_name] = current_idx
            idx_to_class[current_idx] = class_name
            current_idx += 1
        
        class_idx = class_to_idx[class_name]
        
        # 遍历所有子文件夹 (01, 02, ...)
        for subject_dir in class_dir.iterdir():
            if subject_dir.is_dir():
                # 遍历所有的 .npy 文件
                for npy_file in subject_dir.glob("*.npy"):
                    # 将 Path 对象转换为字符串
                    all_filepaths.append(str(npy_file)) 
                    all_labels.append(class_idx)

print(f"扫描完成！总共找到 {len(all_filepaths)} 个特征文件。")
NUM_CLASSES = len(class_to_idx)
print(f"共 {NUM_CLASSES} 个类别: {list(class_to_idx.keys())}")

# 为报告创建类别名称列表 (按索引 0, 1, 2... 排序)
class_names = [idx_to_class[i] for i in range(NUM_CLASSES)]


# --- 第 2 步：划分训练集和验证集 ---
train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_filepaths, 
    all_labels, 
    test_size=0.2, 
    random_state=42, 
    stratify=all_labels # 确保类别比例均衡
)
print(f"训练集样本数: {len(train_paths)}")
print(f"验证集样本数: {len(val_paths)}")


# --- 第 3 步：计算类别权重 (FIX 1) ---
print("\n--- 正在计算类别权重 (用于解决数据不平衡) ---")

# 我们只根据 '训练集' 的标签来计算权重
unique_labels = np.unique(train_labels)
class_weights = compute_class_weight(
    'balanced',
    classes=unique_labels,
    y=train_labels
)

# Keras 需要一个 字典 {class_index: weight}
class_weight_dict = dict(zip(unique_labels, class_weights))
print("计算出的权重如下 (样本越少的类, 权重越高):")
print(class_weight_dict)


# --- 第 4 步：创建 tf.data.Dataset 数据管道 ---
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
train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_ds = val_ds.map(load_npy_file, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

print("\ntf.data 数据管道创建完毕。")


# --- 第 5 步：[修改] 定义一个更深、更复杂的模型 ---
def create_DEEPER_cnn_lstm_model(input_shape, num_classes):
    model = keras.Sequential([
        layers.Input(shape=input_shape), # (30, 126)
        
        # --- 更深的 CNN 特征提取 ---
        layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2), # 序列 30 -> 15
        
        layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2), # 序列 15 -> 7
        
        layers.Conv1D(filters=256, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        
        # --- 更深的 LSTM 序列学习 (使用双向) ---
        layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
        layers.Dropout(0.4),
        layers.Bidirectional(layers.LSTM(128)),
        layers.Dropout(0.4),

        # --- 全连接层 ---
        layers.Dense(256, activation='relu'), 
        layers.Dropout(0.5),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# 使用更深的模型
model = create_DEEPER_cnn_lstm_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
model.summary()


# --- 第 6 步：[修改] 编译模型并设置回调 ---

EPOCHS = 200 # 保持 epochs 足够高，让早停来决定
# [调参] 降低学习率，让模型学得更慢、更精细
LEARNING_RATE = 0.00003

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), # <--- 应用更低的学习率
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

# [修改] 定义 Early Stopping 回调 (增加耐心)
print("\n--- 设置早停 (EarlyStopping) 回调 (用于防止过拟合) ---")
early_stopping_cb = EarlyStopping(
    monitor='val_loss',         # 监控 '验证集损失'
    patience=50,                # <--- [修改] 增加耐心
    restore_best_weights=True,  # 停止时，自动恢复到最佳模型权重
    verbose=1                   # 触发早停时在控制台打印信息
)


# --- 第 7 步：训练模型 (已应用所有调参) ---
print("\n--- 开始训练 (已加入 Class Weights 和 EarlyStopping) ---")
start_time = time.time()

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    class_weight=class_weight_dict,  # <--- [FIX 1] 应用类别权重
    callbacks=[early_stopping_cb]    # <--- [FIX 2] 应用早停回调
)

end_time = time.time()
print(f"--- 训练完成 --- (总耗时: {end_time - start_time:.2f} 秒)")

# [修改] 保存 v3 模型
# 使用 r"..." (raw string) 来确保 Windows 路径正确处理
save_path = r"D:\code\deeplearning\project\DeepLearning-Sign-Language-Recognition\src\Conv1D\sign_language_model_v3_tuned.h5"
model.save(save_path)
print(f"模型已保存为 '{save_path}'")


# --- 第 8 步：生成详细的评估报告 ---
print("\n--- 正在生成详细评估报告 (使用最佳权重) ---")

# 1. 获取所有真实标签 (y_true)
y_true = []
for features, labels in val_ds:
    y_true.extend(labels.numpy())
y_true = np.array(y_true)

# 2. 获取所有预测标签 (y_pred)
predictions_raw = model.predict(val_ds)
y_pred = np.argmax(predictions_raw, axis=1)

# 3. 打印分类报告
print("\n--- 分类报告 (Classification Report) ---")
# (添加 zero_division=0 以防止在极端情况下因 P=0 而报错)
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

# 4. 打印和绘制混淆矩阵
print("\n--- 混淆矩阵 (Confusion Matrix) ---")
cm = confusion_matrix(y_true, y_pred)
print(cm)

print("\n--- 正在绘制混淆矩阵热力图 ---")
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, 
    annot=True,     # 显示数字
    fmt='g',        # 整数格式
    cmap='Blues',   
    xticklabels=class_names, 
    yticklabels=class_names
)
plt.xlabel('Predicted Label (预测标签)', fontsize=13)
plt.ylabel('True Label (真实标签)', fontsize=13)
# [修改] 更新标题
plt.title('Confusion Matrix Heatmap (Tuned Deep Model)', fontsize=15)
plt.show() # 弹出热力图窗口