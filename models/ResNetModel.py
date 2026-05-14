import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, applications

from utils.eval import evaluate_model_on_data
from utils.model_io import find_and_load_model
from utils.plotting import plot_metrics
from utils.wandb_utils import init_wandb_run, finish_wandb_run


class ResNetModel:
    def __init__(self, input_shape=(8, 224, 224, 3), num_classes=2):
        inputs = layers.Input(shape=input_shape)

        self.base_model = applications.ResNet50(input_shape=(224, 224, 3), include_top=False, weights="imagenet")

        x = layers.TimeDistributed(self.base_model)(inputs)
        x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
        x = layers.GlobalAveragePooling1D()(x)

        outputs = layers.Dense(num_classes, activation="softmax")(x)

        self.model = models.Model(inputs, outputs)

    def compile(self, optimizer="adam", loss="sparse_categorical_crossentropy", metrics=None, learning_rate=3e-4):
        if metrics is None:
            metrics = ["accuracy"]

        if optimizer == "adam":
            optimizer = optimizers.Adam(learning_rate=learning_rate)

        model_loss = loss
        if model_loss == "sparse_categorical_crossentropy":
            model_loss = tf.keras.losses.SparseCategoricalCrossentropy()

        self.model.compile(optimizer=optimizer, loss=model_loss, metrics=metrics)

    @staticmethod
    def _prepare_inputs(X):
        X = X.astype("float32")
        if np.max(X) > 1.0:
            X = X / 255.0
        return X

    @classmethod
    def train(
        cls,
        X_train,
        y_train,
        X_val,
        y_val,
        output_dir,
        model_filename,
        epochs,
        label_map,
        train_debugs=None,
        val_debugs=None,
        dataset_name=None,
        cache_label=None,
    ):
        wandb_run = None
        wandb_callback = None

        X_train = cls._prepare_inputs(X_train)
        X_val = cls._prepare_inputs(X_val)

        if X_train.ndim == 4:
            X_train = np.expand_dims(X_train, axis=1)
        if X_val.ndim == 4:
            X_val = np.expand_dims(X_val, axis=1)

        num_classes = len(np.unique(np.concatenate([y_train, y_val])))
        input_shape = X_train.shape[1:]

        model = cls(input_shape=input_shape, num_classes=num_classes)

        wandb_extra_config = {
            "num_classes": int(num_classes),
            "input_shape": tuple(input_shape),
        }
        if cache_label is not None:
            wandb_extra_config["CACHE_VERSION"] = cache_label

        wandb_run, wandb_callback = init_wandb_run(
            model_name=cls.__name__,
            dataset_name=dataset_name,
            epochs=epochs,
            output_dir=output_dir,
            extra_config=wandb_extra_config,
        )

        for layer in model.base_model.layers:
            layer.trainable = False

        model.compile(learning_rate=3e-4, loss="sparse_categorical_crossentropy")

        model.model.summary()

        save_best = callbacks.ModelCheckpoint(
            model_filename, monitor="loss", save_best_only=True, mode="min", verbose=0
        )

        print(f"Starting training ResNetModel for {epochs} epochs...")

        try:
            train_callbacks = [save_best]
            if wandb_callback is not None:
                train_callbacks.append(wandb_callback)

            history = model.model.fit(
                X_train,
                y_train,
                epochs=epochs,
                batch_size=16,
                callbacks=train_callbacks,
            )

            history_dict = history.history

            model.model.load_weights(model_filename)
            model.model.save(model_filename)

            summary_lines = []
            model.model.summary(print_fn=lambda line: summary_lines.append(line))
            model_summary = "\n".join(summary_lines)

            plot_metrics(
                history_dict,
                output_dir,
                model_name=cls.__name__,
                training_debugs=train_debugs,
                validation_debugs=val_debugs,
                dataset_name=dataset_name,
                label_map=label_map,
                cache_label=cache_label,
                model_summary=model_summary,
            )

            return history_dict
        finally:
            finish_wandb_run(wandb_run, model_filename=model_filename)

    def predict(self, images_np):
        images_np = self._prepare_inputs(images_np)
        if images_np.ndim == 4:
            images_np = np.expand_dims(images_np, axis=1)
        return np.argmax(self.model.predict(images_np, verbose=0), axis=1)

    @classmethod
    def eval(cls, val_tuple, label_map=None, dataset_name=None, dataset_path=None, train_tuple=None, cache_label=None):
        model_prefix = cls.__name__.lower()

        loaded_model, model_dir = find_and_load_model(
            model_prefix,
            dataset_name=dataset_name,
            cache_version=cache_label,
        )
        if loaded_model is None:
            print(f"Error: Model {cls.__name__} could not be loaded for evaluation.")
            return

        evaluate_model_on_data(
            loaded_model,
            val_tuple,
            model_dir,
            model_name=cls.__name__,
            label_map=label_map,
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            train_tuple=train_tuple,
            cache_label=cache_label,
        )
