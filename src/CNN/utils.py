# utils.py
import numpy as np

def ensure_dir(path):
    import os
    os.makedirs(path, exist_ok=True)

def labels_to_onehot(labels, num_classes=7):
    import tensorflow as tf
    return tf.keras.utils.to_categorical(labels, num_classes=num_classes)
