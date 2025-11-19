import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 加载模型
MODEL_PATH = r"./sign_language_model_v3_tuned.h5"
model = tf.keras.models.load_model(MODEL_PATH)
print("模型已加载:", MODEL_PATH)

# 数据目录
DATA_DIR = Path(r"../../data/dataset/train")

all_paths = []
all_labels = []
class_to_idx = {}
idx_to_class = {}
cur = 0

# 扫描所有数据
for class_dir in DATA_DIR.iterdir():
    if class_dir.is_dir():
        cname = class_dir.name
        if cname not in class_to_idx:
            class_to_idx[cname] = cur
            idx_to_class[cur] = cname
            cur += 1
        cidx = class_to_idx[cname]
        for sub in class_dir.iterdir():
            if sub.is_dir():
                for f in sub.glob("*.npy"):
                    all_paths.append(str(f))
                    all_labels.append(cidx)

print(f"共 {len(all_paths)} 个测试样本")

# 正确的 class_names 构建方式
class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

# 数据加载
SEQ_LENGTH = 30
INPUT_FEATURES = 126

def load_npy(p):
    d = np.load(p).astype(np.float32)
    return d.reshape(SEQ_LENGTH, INPUT_FEATURES)

X = np.array([load_npy(p) for p in all_paths])
y_true = np.array(all_labels)

# 预测
y_pred = np.argmax(model.predict(X), axis=1)

# 输出报告
print(classification_report(y_true, y_pred, target_names=class_names))

# 混淆矩阵
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.title("Confusion Matrix (All Data Test)")
plt.show()
