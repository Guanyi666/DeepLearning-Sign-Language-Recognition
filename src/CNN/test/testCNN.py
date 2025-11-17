import os
import glob
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "sign_mobilenet_custom.h5")

DATA_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../../../data/dataset/train"))
VISUALIZE_EACH = False     # 是否对每个样本都显示可视化（True/False）
SAMPLES_PER_CLASS = 10      # 每类随机选多少个样本


# ----------------------------------------------------
# 字典
# ----------------------------------------------------
ID_TO_WORD = {
    0: "none",
    1: "good",
    2: "he",
    3: "moring",
    4: "thankyou",
    5: "very",
    6: "you"
}
WORD_TO_ID = {v: k for k, v in ID_TO_WORD.items()}


# ----------------------------------------------------
# 加载某类的所有 npy
# ----------------------------------------------------
def load_samples(word):
    pattern = os.path.join(DATA_ROOT, word, "*", "*_features.npy")
    paths = glob.glob(pattern)
    samples = []
    for p in paths:
        try:
            arr = np.load(p)
            if arr.shape == (30, 126):
                samples.append(arr)
        except:
            pass
    return samples


# ----------------------------------------------------
# Tester 类
# ----------------------------------------------------
class SignWordTesterMulti:

    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型未找到: {model_path}")

        print("加载模型:", model_path)
        self.model = tf.keras.models.load_model(model_path)
        print("模型加载成功！")

    def preprocess(self, x):
        x = x.astype(np.float32)
        m, s = x.mean(), (x.std() if x.std() > 1e-6 else 1)
        x = (x - m) / s
        x = np.expand_dims(x, -1)
        x = np.repeat(x, 3, -1)
        return np.expand_dims(x, 0)

    def predict(self, sample):
        x = self.preprocess(sample)
        r, l = self.model.predict(x, verbose=0)
        pred_id = np.argmax(r[0])
        conf = np.max(r[0])
        return pred_id, conf

    # --------------------------
    # 可视化
    # --------------------------
    def visualize(self, sample, true_label, pred_id, conf):
        plt.figure(figsize=(16, 6))

        # Heatmap
        plt.subplot(1, 3, 1)
        plt.imshow(sample, aspect='auto')
        plt.title(f"Heatmap\nTrue={true_label}")
        plt.colorbar()

        # 前5维时间序列
        plt.subplot(1, 3, 2)
        for i in range(5):
            plt.plot(sample[:, i])
        plt.title("Time Series")

        # 预测柱状图
        plt.subplot(1, 3, 3)
        ids = list(ID_TO_WORD.keys())
        words = list(ID_TO_WORD.values())
        probs = np.zeros(len(words))
        probs[pred_id] = conf

        plt.bar(words, probs)
        plt.xticks(rotation=45)
        plt.title(f"Pred={ID_TO_WORD[pred_id]} ({conf:.2f})")
        plt.tight_layout()
        plt.show()

    # --------------------------
    # 多样本评估（随机抽样）
    # --------------------------
    def evaluate_multi(self, samples_per_class=SAMPLES_PER_CLASS):
        y_true = []
        y_pred = []

        for wid, word in ID_TO_WORD.items():
            if word == "none":
                continue

            print(f"\n===== 测试词汇：{word} =====")
            samples = load_samples(word)

            if len(samples) == 0:
                print(f"⚠ 没有数据: {word}")
                continue

            # 随机取指定数量的样本
            chosen = random.sample(samples, min(samples_per_class, len(samples)))

            for sample in chosen:
                pred_id, conf = self.predict(sample)

                print(f"真实: {word:<10} → 预测: {ID_TO_WORD[pred_id]:<10}  ({conf:.2f})")

                y_true.append(WORD_TO_ID[word])
                y_pred.append(pred_id)

                if VISUALIZE_EACH:
                    self.visualize(sample, word, pred_id, conf)

        # --------------------------
        # 整体准确率
        # --------------------------
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        acc = (y_true == y_pred).mean()
        print("\n====================================")
        print(f"整体测试准确率: {acc:.3f}")
        print("====================================")

        # --------------------------
        # 混淆矩阵
        # --------------------------
        cm = confusion_matrix(y_true, y_pred, labels=list(ID_TO_WORD.keys()))
        disp = ConfusionMatrixDisplay(cm, display_labels=list(ID_TO_WORD.values()))
        disp.plot(xticks_rotation='vertical', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.show()


# ----------------------------------------------------
# Run
# ----------------------------------------------------
if __name__ == "__main__":
    tester = SignWordTesterMulti(MODEL_PATH)
    tester.evaluate_multi(samples_per_class=SAMPLES_PER_CLASS)
