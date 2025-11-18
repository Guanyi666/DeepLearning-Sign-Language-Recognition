# evaluate.py - fixed dynamic label handling
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from dataset import build_dataset
from model import build_rnn_lstm_model, compile_model

LABEL_NAMES = {
    0: "not_detected",
    1: "good",
    2: "he",
    3: "morning",
    4: "thank_you",
    5: "very",
    6: "you"
}

def load_model_auto():
    base = os.path.dirname(__file__)
    model_path = os.path.join(base, "test", "best_model.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    print(f"Loading model: {model_path}")
    return tf.keras.models.load_model(model_path)

def load_full_dataset(data_root):
    print("Loading ALL train data...")
    X, R, L, P = build_dataset(data_root, split="train", shuffle=False)
    total = len(X)

    test_dir = os.path.join(data_root, "test")
    if os.path.exists(test_dir):
        print("Loading test data...")
        X2, R2, L2, P2 = build_dataset(data_root, split="test", shuffle=False)
        X += X2
        R += R2
        P += P2
        total += len(X2)

    print(f"Total samples loaded = {total}")
    return np.stack(X), np.array(R), P

def evaluate_all(data_root):
    X, y_true, paths = load_full_dataset(data_root)
    model = load_model_auto()

    pred_right, _ = model.predict(X)
    y_pred = np.argmax(pred_right, axis=1)

    # ----------------------------------
    #  1. 自动检测实际存在的类别
    # ----------------------------------
    unique_classes = sorted(list(set(y_true)))
    print(f"\nDetected real classes in dataset: {unique_classes}\n")

    class_names = [LABEL_NAMES[c] for c in unique_classes]

    # ----------------------------------
    #  2. Overall Accuracy
    # ----------------------------------
    acc = np.mean(y_pred == y_true)
    print("============================")
    print(f"   🎯 OVERALL ACCURACY = {acc:.4f}")
    print("============================\n")

    # ----------------------------------
    #  3. Confusion Matrix
    # ----------------------------------
    print("📌 Confusion Matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
    print(cm)

    # ----------------------------------
    #  4. Classification Report
    # ----------------------------------
    print("\n📌 Classification Report:")
    print(classification_report(
        y_true,
        y_pred,
        labels=unique_classes,
        target_names=class_names
    ))

    # ----------------------------------
    #  5. Example predictions
    # ----------------------------------
    print("\n===== Example Predictions (first 20) =====")
    for i in range(min(20, len(X))):
        print(f"[{i}] {paths[i]}")
        print(f"     True: {LABEL_NAMES[y_true[i]]}")
        print(f"     Pred: {LABEL_NAMES[y_pred[i]]}\n")

    return acc


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="../../data/dataset")
    args = parser.parse_args()
    evaluate_all(args.data_root)
