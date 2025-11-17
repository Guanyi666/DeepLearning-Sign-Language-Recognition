from data_loader import load_all_data
from preprocess import build_dataset
from model_builder import build_model
from trainer import train_model


def main():
    X_train, X_val, yr_train, yr_val, yl_train, yl_val = load_all_data()

    train_ds = build_dataset(X_train, yr_train, yl_train, training=True)
    val_ds = build_dataset(X_val, yr_val, yl_val, training=False)

    model = build_model()
    model.summary()

    train_model(model, train_ds, val_ds)


if __name__ == "__main__":
    main()
