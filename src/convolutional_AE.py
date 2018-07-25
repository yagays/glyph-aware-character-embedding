"""
Usage:
   $ python src/convolutional_AE.py --img_dir /path/to/NotoSansCJKjp-Regular --n_embedding 300
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape
from keras.preprocessing.image import load_img, img_to_array
from keras.callbacks import TensorBoard, EarlyStopping


def CAE(input_shape=(60, 60, 1), filters=[32, 64, 128, 10]):
    # https://github.com/XifengGuo/DCEC/blob/master/ConvAE.py
    model = Sequential()
    if input_shape[0] % 8 == 0:
        pad3 = "same"
    else:
        pad3 = "valid"
    model.add(Conv2D(filters[0], 5, strides=2, padding="same",
                     activation="relu", name="conv1", input_shape=input_shape))
    model.add(Conv2D(filters[1], 5, strides=2, padding="same", activation="relu", name="conv2"))
    model.add(Conv2D(filters[2], 3, strides=2, padding=pad3, activation="relu", name="conv3"))

    model.add(Flatten())
    model.add(Dense(units=filters[3], name="embedding"))
    model.add(Dense(units=filters[2] * int(input_shape[0] / 8) * int(input_shape[0] / 8), activation="relu"))

    model.add(Reshape((int(input_shape[0] / 8), int(input_shape[0] / 8), filters[2])))
    model.add(Conv2DTranspose(filters[1], 3, strides=2, padding=pad3, activation="relu", name="deconv3"))

    model.add(Conv2DTranspose(filters[0], 5, strides=2, padding="same", activation="relu", name="deconv2"))

    model.add(Conv2DTranspose(input_shape[2], 5, strides=2, padding="same", name="deconv1"))
    model.summary()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convolutional AutoEncoder")
    parser.add_argument("--img_dir", type=str)
    parser.add_argument("--n_embedding", type=int)
    args = parser.parse_args()

    img_path = Path(args.img_dir)
    tf_log_dir = ".log/"
    exp_stamp = f"convolutional_AE_{args.n_embedding}_{img_path.stem}"

    X = []
    label = []
    for i, img_file in enumerate(img_path.glob("*.png")):
        temp_img = load_img(img_file, target_size=(60, 60))
        img_array = img_to_array(temp_img.convert("L"))
        img_array = img_array.astype("float32") / 255.
        img_array = img_array.reshape(60, 60)
        X.append(img_array)
        label.append(img_file.stem.replace("char_", ""))
    X = np.array(X)

    X_train, X_test = train_test_split(X, test_size=0.10, random_state=42)
    X_train = np.reshape(X_train, (len(X_train), 60, 60, 1))
    X_test = np.reshape(X_test, (len(X_test), 60, 60, 1))

    model = CAE(input_shape=(60, 60, 1), filters=[16, 64, 128, args.n_embedding])
    model.compile(optimizer="adam", loss="mse")

    callback_earlystopping = EarlyStopping(monitor="val_loss", patience=3, verbose=0, mode="auto")
    callback_tensorboard = TensorBoard(log_dir=tf_log_dir)

    history = model.fit(X_train, X_train,
                        epochs=100,
                        batch_size=256,
                        shuffle=True,
                        validation_data=(X_test, X_test),
                        callbacks=[callback_earlystopping, callback_tensorboard])

    model.save(f"model/{exp_stamp}.h5")

    X = np.reshape(X, (len(X), 60, 60, 1))
    feature_model = Model(inputs=model.input, outputs=model.get_layer(name="embedding").output)
    features = feature_model.predict(X)

    df = pd.DataFrame(features)
    df["codepoint"] = [int(i) for i in label]
    df["char"] = [chr(int(i)) for i in label]
    df.sort_values("codepoint", inplace=True)

    with open(f"data/{exp_stamp}.txt", "w") as f:
        f.write(f"{len(label)} {args.n_embedding}\n")
        line = df.to_string(
            index=None,
            header=None,
            columns=["char"] + [i for i in range(args.n_embedding)],
            float_format="%.6f")
        f.write(line.replace("  ", " ") + "\n")
