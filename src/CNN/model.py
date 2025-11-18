# model.py
import tensorflow as tf

def build_rnn_lstm_model(input_shape=(30, 126), num_classes=7, dropout=0.3):
    """
    Build a model with shared encoder (LSTM) and two softmax heads (right_out, left_out).
    Right_out: predicts 0..6 (7 classes)
    Left_out: same shape (left data currently all zeros, but allows future training)
    """
    inp = tf.keras.Input(shape=input_shape, name="input_features")
    # optional projection
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(128, activation="relu"))(inp)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128))(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    shared = tf.keras.layers.Dense(128, activation="relu")(x)

    # right hand head
    right_dense = tf.keras.layers.Dense(64, activation="relu")(shared)
    right_out = tf.keras.layers.Dense(num_classes, activation="softmax", name="right_out")(right_dense)

    # left hand head
    left_dense = tf.keras.layers.Dense(64, activation="relu")(shared)
    left_out = tf.keras.layers.Dense(num_classes, activation="softmax", name="left_out")(left_dense)

    model = tf.keras.Model(inputs=inp, outputs=[right_out, left_out], name="rnn_lstm_hands")
    return model

def compile_model(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss={
            "right_out": "categorical_crossentropy",
            "left_out": "categorical_crossentropy"
        },
        loss_weights={"right_out": 1.0, "left_out": 0.5},  # left less important now
        metrics={"right_out": "accuracy", "left_out": "accuracy"}
    )
    return model
