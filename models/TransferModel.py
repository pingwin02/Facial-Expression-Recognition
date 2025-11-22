import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, applications

from utils.eval import evaluate_model_on_data
from utils.model_io import find_and_load_model
from utils.plotting import plot_metrics


class TransferModel:
    """A model using MobileNetV2 for facial expression classification."""

    def __init__(self, input_shape=(48, 48, 1), num_classes=7):
        """Initialize the Model using MobileNetV2 as a base.

        Args:
            input_shape (tuple): Shape of a single input image.
            num_classes (int): Number of output classes.
        """
        inputs = layers.Input(shape=input_shape)

        x = layers.Conv2D(3, (1, 1), padding='same')(inputs)

        base_model = applications.MobileNetV2(
            input_shape=(input_shape[0], input_shape[1], 3),
            include_top=False,
            weights='imagenet',
            alpha=1.0
        )

        base_model.trainable = True

        x = base_model(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)

        outputs = layers.Dense(num_classes, activation="softmax")(x)

        self.model = models.Model(inputs, outputs)

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
            optimizer = optimizers.Adam(learning_rate=0.0001)

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    @classmethod
    def train(cls, X_train, y_train, X_val, y_val, output_dir, epochs):
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

        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )

        history = model.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr]
        )

        model_filename = f"{cls.__name__}_model.keras"
        model.model.save(os.path.join(output_dir, model_filename))
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
