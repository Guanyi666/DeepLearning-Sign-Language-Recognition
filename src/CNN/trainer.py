import tensorflow as tf

MODEL_SAVE_PATH = "test/sign_mobilenet_custom.h5"


def train_model(model, train_ds, val_ds, epochs=40):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss={"right": "categorical_crossentropy", "left": "categorical_crossentropy"},
        loss_weights={"right": 1.0, "left": 0.5},
        metrics={"right": "accuracy", "left": "accuracy"}
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint("best_" + MODEL_SAVE_PATH, save_best_only=True),
        tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

    model.save(MODEL_SAVE_PATH)
    print("模型已保存:", MODEL_SAVE_PATH)

    return model
