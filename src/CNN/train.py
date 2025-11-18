# train.py
import os
import argparse
import numpy as np
import tensorflow as tf
from dataset import build_dataset, generator_from_lists, NUM_CLASSES
from model import build_rnn_lstm_model, compile_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="../../data/dataset", help="root dataset dir")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--model_dir", type=str, default="checkpoints")
    parser.add_argument("--save_best_only", action="store_true", default=True)
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)

    print("Loading train data...")
    train_X, train_r, train_l, train_paths = build_dataset(args.data_root, split="train", shuffle=True)
    total_train_samples = len(train_X)
    print(f"Train samples (total): {total_train_samples}")

    print("Attempting to load validation data from test/ ...")
    test_path = os.path.join(args.data_root, "test")

    # ------------------------------------------------------
    # CASE 1 — test/ exists → use it as validation
    # ------------------------------------------------------
    if os.path.exists(test_path):
        print("test/ folder FOUND. Using test set as validation.")
        val_X, val_r, val_l, _ = build_dataset(args.data_root, split="test", shuffle=False)

    else:
        # ------------------------------------------------------
        # CASE 2 — test/ missing → split train as validation
        # ------------------------------------------------------
        print("test/ folder NOT found. Splitting train data into train/validation sets...")

        val_size = max(20, int(total_train_samples * 0.2))  # 20%
        print(f"Validation size = {val_size}")

        val_X = train_X[:val_size]
        val_r = train_r[:val_size]
        val_l = train_l[:val_size]

        train_X = train_X[val_size:]
        train_r = train_r[val_size:]
        train_l = train_l[val_size:]

        print(f"Actual train size = {len(train_X)}")
        print(f"Actual validation size = {len(val_X)}")

    # numpy conversion for validation set
    val_X_np = np.stack(val_X, axis=0)
    val_yr_np = tf.keras.utils.to_categorical(np.array(val_r, dtype=np.int32), num_classes=NUM_CLASSES)
    val_yl_np = tf.keras.utils.to_categorical(np.array(val_l, dtype=np.int32), num_classes=NUM_CLASSES)

    # ------------------------------------------------------
    # Build & compile model
    # ------------------------------------------------------
    model = build_rnn_lstm_model(input_shape=(30, 126), num_classes=NUM_CLASSES)
    model = compile_model(model)
    model.summary()

    # ------------------------------------------------------
    # Callbacks — FIXED version
    # ------------------------------------------------------
    checkpoint_path = os.path.join(args.model_dir, "best_model.keras")

    cb_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_right_out_accuracy",
            save_best_only=args.save_best_only,
            mode="max",        # <—— FIXED
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_right_out_accuracy",
            patience=4,
            factor=0.5,
            mode="max",        # <—— FIXED
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_right_out_accuracy",
            patience=8,
            restore_best_weights=True,
            mode="max",        # <—— FIXED
            verbose=1
        )
    ]

    # ------------------------------------------------------
    # Training
    # ------------------------------------------------------
    train_gen = generator_from_lists(train_X, train_r, train_l, batch_size=args.batch_size)
    steps_per_epoch = max(1, len(train_X) // args.batch_size)

    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs,
        validation_data=(val_X_np, {"right_out": val_yr_np, "left_out": val_yl_np}),
        callbacks=cb_list
    )

    # ------------------------------------------------------
    # Save model into src/CNN/test/
    # ------------------------------------------------------
    save_dir = os.path.join(os.path.dirname(__file__), "test")
    os.makedirs(save_dir, exist_ok=True)

    best_path = os.path.join(save_dir, "best_model.keras")
    final_path = os.path.join(save_dir, "final_model.keras")

    # save best model (copied from checkpoint)
    if os.path.exists(os.path.join(args.model_dir, "best_model.keras")):
        import shutil
        shutil.copy(os.path.join(args.model_dir, "best_model.keras"), best_path)
        print(f"Copied best model to: {best_path}")

    # save final trained model
    model.save(final_path)
    print(f"Final model saved to: {final_path}")

if __name__ == "__main__":
    main()
