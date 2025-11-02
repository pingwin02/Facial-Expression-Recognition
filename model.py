import os

import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv3D,
    MaxPool3D,
    ConvLSTM2D,
    Flatten,
    Dense,
)
from tensorflow.keras.models import Model

MODEL_DIR = "models"
MODEL_FILENAME = "hybrid_vfer_model.keras"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)


def build_hybrid_vfer_model(time_steps, img_height, img_width, channels, num_classes):
    """
    Build and compile the hybrid 3D-CNN + ConvLSTM model for video facial expression recognition.

    Args:
        time_steps (int): Number of frames in the input clip.
        img_height (int): Height of each frame.
        img_width (int): Width of each frame.
        channels (int): Number of channels per frame.
        num_classes (int): Number of output classes.

    Returns:
        tensorflow.keras.Model: Compiled Keras model ready for training or inference.
    """
    input_shape = (time_steps, img_height, img_width, channels)
    input_layer = Input(shape=input_shape, name="input_video_frames")

    x = Conv3D(
        filters=64,
        kernel_size=(3, 3, 3),
        activation="relu",
        padding="same",
        name="conv3d_1_64f",
    )(input_layer)
    x = Conv3D(
        filters=64,
        kernel_size=(3, 3, 3),
        activation="relu",
        padding="same",
        name="conv3d_2_64f",
    )(x)
    x = MaxPool3D(pool_size=(2, 2, 2), name="maxpool3d_1")(x)

    x = Conv3D(
        filters=128,
        kernel_size=(3, 3, 3),
        activation="relu",
        padding="same",
        name="conv3d_3_128f",
    )(x)
    x = MaxPool3D(pool_size=(2, 2, 2), name="maxpool3d_2")(x)

    x = ConvLSTM2D(
        filters=16,
        kernel_size=(5, 5),
        padding="same",
        return_sequences=False,
        name="convlstm_16_units_5x5",
    )(x)

    x = Flatten(name="flatten_output")(x)
    x = Dense(units=20, activation="relu", name="fully_connected_layer_20")(x)
    output_layer = Dense(
        units=num_classes, activation="softmax", name="softmax_output"
    )(x)

    model = Model(
        inputs=input_layer,
        outputs=output_layer,
        name="Hybrid_3DCNN_ConvLSTM_VFER_PDF",
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_vfer_model(model: Model, path: str):
    """
    Save the provided Keras model to the specified filesystem path, creating directories if needed.

    Args:
        model (tensorflow.keras.Model): The model to save.
        path (str): Filesystem path where the model will be saved.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        model.save(path)
        print(f"INFO: Model successfully saved to: {path}")
    except Exception as e:
        print(f"ERROR: Could not save model: {e}")


def load_vfer_model(path: str) -> Model | None:
    """
    Load a Keras model from the given path if it exists.

    Args:
        path (str): Filesystem path to the saved model.

    Returns:
        tensorflow.keras.Model | None: The loaded model, or None if loading failed or file does not exist.
    """
    if not os.path.exists(path):
        print(f"INFO: Model file does not exist at: {path}")
        return None
    try:
        model = tf.keras.models.load_model(path)
        print(f"INFO: Model successfully loaded from: {path}")
        return model
    except Exception as e:
        print(f"ERROR: Could not load model: {e}")
        return None
