import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models, callbacks, optimizers

from utils.eval import evaluate_model_on_data
from utils.model_io import find_and_load_model
from utils.plotting import plot_metrics


class BinaryModel:
    def __init__(self, input_shape=(48, 48, 2)):
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
                layers.Dense(1, activation="sigmoid"),
            ]
        )

    def compile(self, optimizer="adam", loss="binary_crossentropy", metrics=None):
        if metrics is None:
            metrics = ["binary_accuracy"]

        if optimizer == "adam":
            optimizer = optimizers.Adam(learning_rate=0.001)

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    @classmethod
    def train(cls, X_train, y_train, X_val, y_val, output_dir, model_filename, epochs, label_map):
        if hasattr(X_train, "ndim") and X_train.ndim == 5:
            center_idx = X_train.shape[1] // 2
            X_train = X_train[:, center_idx]
        if hasattr(X_val, "ndim") and X_val.ndim == 5:
            center_idx = X_val.shape[1] // 2
            X_val = X_val[:, center_idx]

        if "neutral" in label_map:
            target_idx = label_map["neutral"]
        else:
            raise ValueError("Label map must contain 'neutral' for binary classification.")

        y_train_bin = (y_train == target_idx).astype("float32")
        y_val_bin = (y_val == target_idx).astype("float32")

        classes = np.unique(y_train_bin)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_bin)
        class_weights = dict(zip(classes, weights))

        model = cls()
        model.compile()

        model.model.summary()

        reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=0)

        history = model.model.fit(
            X_train,
            y_train_bin,
            validation_data=(X_val, y_val_bin),
            epochs=epochs,
            batch_size=64,
            callbacks=[reduce_lr],
            class_weight=class_weights,
        )

        model.model.save(model_filename)
        plot_metrics(history.history, output_dir, model_name=cls.__name__)
        return history

    def predict(self, images_np):
        if hasattr(images_np, "ndim") and images_np.ndim == 5:
            center_idx = images_np.shape[1] // 2
            images_np = images_np[:, center_idx]
        return (self.model.predict(images_np, verbose=0) > 0.5).astype("int32").flatten()

    @classmethod
    def eval(cls, val, output_dir, label_map=None, dataset_name=None, dataset_path=None, train_tuple=None):
        model_prefix = cls.__name__.lower()
        loaded_model = find_and_load_model(model_prefix)
        if loaded_model is None:
            return

        evaluate_model_on_data(
            loaded_model,
            val,
            output_dir,
            model_name=cls.__name__,
            label_map=label_map,
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            train_tuple=train_tuple,
        )
