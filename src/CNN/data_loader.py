import os
import glob
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

DATA_ROOT = "../../data/dataset"
WORD_TO_ID = {
    "good": 1,
    "he": 2,
    "moring": 3,
    "thankyou": 4,
    "very": 5,
    "you": 6
}
NUM_CLASSES = 7


def _load_word_samples(root_dir, word_name):
    samples = []
    pattern = os.path.join(root_dir, "train", word_name, "*", "*_features.npy")

    for path in glob.glob(pattern):
        try:
            arr = np.load(path)
            if arr.shape == (30, 126):
                samples.append(arr.astype(np.float32))
        except:
            continue

    return samples


def load_all_data(root_dir=DATA_ROOT):
    X = []
    y_right = []
    y_left = []

    for word, wid in WORD_TO_ID.items():
        samples = _load_word_samples(root_dir, word)
        for s in samples:
            X.append(s)
            y_right.append(wid)
            y_left.append(0)  # 左手全 0

    X = np.array(X)
    y_right = tf.keras.utils.to_categorical(y_right, num_classes=NUM_CLASSES)
    y_left = tf.keras.utils.to_categorical(y_left, num_classes=NUM_CLASSES)

    return train_test_split(X, y_right, y_left, test_size=0.2, stratify=y_right.argmax(1))
