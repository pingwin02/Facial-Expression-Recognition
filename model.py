import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPool3D, ConvLSTM2D, Flatten, Dense
import os
from data_processor import TIME_STEPS, IMG_HEIGHT, IMG_WIDTH, CHANNELS, NUM_CLASSES

MODEL_DIR = "models"
MODEL_FILENAME = "hybrid_vfer_model.keras"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)


def build_hybrid_vfer_model(time_steps, img_height, img_width, channels, num_classes):
    """Implements the hybrid 3D-CNN and ConvLSTM architecture."""
    input_shape = (time_steps, img_height, img_width, channels)
    input_layer = Input(shape=input_shape, name="input_video_frames")

    # 3D-CNN Block
    x = Conv3D(filters=32, kernel_size=(3, 3, 3), activation="relu", padding="same", name="conv3d_1")(input_layer)
    x = Conv3D(filters=64, kernel_size=(3, 3, 3), activation="relu", padding="same", name="conv3d_2")(x)
    x = Conv3D(filters=128, kernel_size=(3, 3, 3), activation="relu", padding="same", name="conv3d_3")(x)
    x = MaxPool3D(pool_size=(1, 2, 2), name="maxpool3d_1")(x)

    # ConvLSTM Block
    x = ConvLSTM2D(filters=16, kernel_size=(3, 3), padding="same", return_sequences=False, name="convlstm_16_units")(x)

    # Classifier
    x = Flatten(name="flatten_output")(x)
    x = Dense(units=128, activation="relu", name="fully_connected_layer")(x)
    output_layer = Dense(units=num_classes, activation="softmax", name="softmax_output")(x)

    model = Model(inputs=input_layer, outputs=output_layer, name="Hybrid_3DCNN_ConvLSTM_VFER")

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def save_vfer_model(model: Model, path: str):
    """Saves the trained Keras model to the specified path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        model.save(path)
        print(f"INFO: Model successfully saved to: {path}")
    except Exception as e:
        print(f"ERROR: Could not save model: {e}")


def load_vfer_model(path: str) -> Model | None:
    """Loads a Keras model from the specified path."""
    if not os.path.exists(path):
        return None
    try:
        model = tf.keras.models.load_model(path)
        print(f"INFO: Model successfully loaded from: {path}")
        return model
    except Exception as e:
        print(f"ERROR: Could not load model: {e}")
        return None
