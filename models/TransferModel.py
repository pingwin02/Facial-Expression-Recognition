import numpy as np
from tensorflow.keras import layers, models, callbacks, optimizers, applications

from utils.eval import evaluate_model_on_data
from utils.model_io import find_and_load_model
from utils.plotting import plot_metrics


class TransferModel:
    def __init__(self, input_shape=(48, 48, 4), num_classes=7):
        inputs = layers.Input(shape=input_shape)

        x = layers.Conv2D(3, (3, 3), padding="same", use_bias=False)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        base_model = applications.MobileNetV2(
            input_shape=(input_shape[0], input_shape[1], 3), include_top=False, weights="imagenet", alpha=1.0
        )

        base_model.trainable = True

        x = base_model(x)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)

        x = layers.Dense(256)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(0.5)(x)

        outputs = layers.Dense(num_classes, activation="softmax")(x)

        self.model = models.Model(inputs, outputs)

    def compile(self, optimizer="adam", loss="sparse_categorical_crossentropy", metrics=None):
        if metrics is None:
            metrics = ["accuracy"]

        if optimizer == "adam":
            optimizer = optimizers.Adam(learning_rate=1e-4)

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    @classmethod
    def train(cls, X_train, y_train, X_val, y_val, output_dir, model_filename, epochs, label_map):
        num_classes = len(np.unique(y_train))

        model = cls(num_classes=num_classes)
        model.compile()

        model.model.summary()

        early_stopping = callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1)

        reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=4, min_lr=1e-7, verbose=1)

        print(f"Starting training TransferModel for {epochs} epochs...")

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
        return np.argmax(self.model.predict(images_np, verbose=0), axis=1)

    @classmethod
    def eval(cls, val_tuple, output_dir, label_map=None, dataset_name=None):
        model_prefix = cls.__name__.lower()

        loaded_model = find_and_load_model(model_prefix)
        if loaded_model is None:
            print(f"Error: Model {cls.__name__} could not be loaded for evaluation.")
            return

        evaluate_model_on_data(
            loaded_model,
            val_tuple,
            output_dir,
            model_name=cls.__name__,
            label_map=label_map,
            dataset_name=dataset_name,
        )
