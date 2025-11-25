import os

import numpy as np
from tensorflow.keras import layers, models, callbacks, optimizers

from utils.eval import evaluate_model_on_data
from utils.model_io import find_and_load_model
from utils.plotting import plot_metrics


class SimpleModel:
    """A minimal convolutional Keras model for facial expression classification."""

    def __init__(self, input_shape=(48, 48, 1), num_classes=7):
        """Initialize the Keras Sequential model.

        Args:
            input_shape (tuple): Shape of a single input image.
            num_classes (int): Number of output classes.
        """
        self.model = models.Sequential(
            [
                layers.Conv2D(32, (3, 3), padding="same", input_shape=input_shape),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                layers.Conv2D(64, (3, 3), padding="same"),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                layers.Conv2D(128, (3, 3), padding="same"),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                layers.Flatten(),
                layers.Dense(128),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation="softmax"),
            ]
        )

    def compile(self, optimizer="adam", loss="sparse_categorical_crossentropy", metrics=None):
        """Compile the Keras model with given optimizer, loss and metrics.

        Args:
            optimizer (str|tf.keras.Optimizer): Optimizer to use.
            loss (str): Loss function name.
            metrics (list): List of metric names or callables.
        """
        if metrics is None:
            metrics = ["accuracy"]

        if optimizer == "adam":
            optimizer = optimizers.Adam(learning_rate=0.001)

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    @classmethod
    def train(cls, X_train, y_train, X_val, y_val, output_dir, model_filename, epochs):
        """Train the model and save trained weights and metrics.

        Args:
            X_train, y_train: Training data and labels.
            X_val, y_val: Validation data and labels.
            output_dir (str): Directory to save model and plots.
            epochs (int): Number of training epochs.

        Returns:
            History object from Keras fit.
        """
        num_classes = len(np.unique(y_train))
        model = cls(num_classes=num_classes)
        model.compile()

        early_stopping = callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1)

        reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1)

        history = model.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=64,
            callbacks=[early_stopping, reduce_lr],
        )

        model.model.save(model_filename)
        plot_metrics(history.history, output_dir, model_name=cls.__name__)
        return history

    def predict(self, images_np):
        """Return predicted class indices for a batch of images."""
        return np.argmax(self.model.predict(images_np), axis=1)

    @classmethod
    def eval(cls, val, output_dir, label_map=None, dataset_name=None):
        """Load a trained model and evaluate it on validation data.

        Args:
            val: Validation tuple (X_val, y_val, debugs).
            output_dir (str): Directory where outputs will be written.
            label_map (dict): Optional mapping from label name->int.
            dataset_name (str): Optional dataset identifier for plots.
        """
        model_prefix = cls.__name__.lower()
        loaded_model = find_and_load_model(model_prefix)
        if loaded_model is None:
            print("Error: Model could not be loaded for evaluation.")
            return
        evaluate_model_on_data(
            loaded_model,
            val,
            output_dir,
            model_name=cls.__name__,
            label_map=label_map,
            dataset_name=dataset_name,
        )
