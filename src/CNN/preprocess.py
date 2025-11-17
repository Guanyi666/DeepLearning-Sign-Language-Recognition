import tensorflow as tf

BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE


def preprocess_sample(x):
    x = tf.cast(x, tf.float32)
    mean, std = tf.reduce_mean(x), tf.math.reduce_std(x)
    std = tf.where(std == 0.0, 1.0, std)
    x = (x - mean) / std

    x = tf.expand_dims(x, axis=-1)
    x = tf.tile(x, [1, 1, 3])
    return x


def build_dataset(X, yr, yl, training=True):
    ds = tf.data.Dataset.from_tensor_slices((X, {"right": yr, "left": yl}))
    if training:
        ds = ds.shuffle(buffer_size=len(X))
    ds = ds.map(lambda x, y: (preprocess_sample(x), y), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds
