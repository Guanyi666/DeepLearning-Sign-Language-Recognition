from tensorflow.keras import layers, models


def inverted_residual_block(x, filters, expansion, stride):
    in_c = x.shape[-1]

    expanded = layers.Conv2D(in_c * expansion, 1, padding='same', activation='relu')(x)
    expanded = layers.BatchNormalization()(expanded)

    dw = layers.DepthwiseConv2D(3, strides=stride, padding='same', activation='relu')(expanded)
    dw = layers.BatchNormalization()(dw)

    projected = layers.Conv2D(filters, 1, padding='same')(dw)
    projected = layers.BatchNormalization()(projected)

    se = layers.GlobalAveragePooling2D()(projected)
    se = layers.Dense(filters // 8, activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, filters))(se)
    se_out = layers.Multiply()([projected, se])

    if stride == 1 and in_c == filters:
        se_out = layers.Add()([x, se_out])

    return se_out


def build_model(input_shape=(30, 126, 3), num_classes=7):
    inp = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inp)
    x = layers.BatchNormalization()(x)

    x = inverted_residual_block(x, 32, expansion=2, stride=1)
    x = inverted_residual_block(x, 32, expansion=2, stride=1)

    x = inverted_residual_block(x, 64, expansion=2, stride=2)
    x = inverted_residual_block(x, 64, expansion=2, stride=1)

    x = inverted_residual_block(x, 96, expansion=2, stride=2)
    x = inverted_residual_block(x, 96, expansion=2, stride=1)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)

    right = layers.Dense(128, activation='relu')(x)
    right = layers.Dense(num_classes, activation='softmax', name="right")(right)

    left = layers.Dense(64, activation='relu')(x)
    left = layers.Dense(num_classes, activation='softmax', name="left")(left)

    return models.Model(inp, [right, left])
