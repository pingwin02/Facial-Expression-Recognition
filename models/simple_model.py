import tensorflow as tf
from tensorflow.keras import layers, models


def SimpleModel(input_shape=(48, 48, 1), num_classes=7):
    model = models.Sequential(
        [
            layers.Conv2D(16, (3, 3), padding="same", activation="relu", input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model
