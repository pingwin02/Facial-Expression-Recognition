import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, applications

from models._base import BaseModel
from utils.eval import evaluate_model_on_data
from utils.model_io import find_and_load_model
from utils.plotting import plot_metrics
from utils.wandb_utils import init_wandb_run, finish_wandb_run


class ResNetModel(BaseModel):
    def __init__(self, input_shape=(8, 224, 224, 3), num_classes=2):
        inputs = layers.Input(shape=input_shape)

        data_augmentation = models.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomContrast(0.2),
                layers.RandomBrightness(0.15),
                layers.RandomRotation(0.08),
                layers.RandomZoom(height_factor=(-0.1, 0.15), width_factor=(-0.1, 0.15)),
            ],
            name="frame_aug",
        )

        x = layers.TimeDistributed(data_augmentation, name="video_aug")(inputs)
        x = layers.TimeDistributed(layers.Rescaling(scale=2.0, offset=-1.0))(x)

        self.base_model = applications.ResNet50(input_shape=(224, 224, 3), include_top=False, weights="imagenet")

        x = layers.TimeDistributed(self.base_model)(x)
        x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
        x = layers.GlobalAveragePooling1D()(x)

        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        outputs = layers.Dense(num_classes, activation="softmax")(x)

        self.model = models.Model(inputs, outputs)

    def set_fine_tune_layers(self, trainable_backbone_layers):
        trainable_backbone_layers = max(0, int(trainable_backbone_layers))
        total_layers = len(self.base_model.layers)

        if trainable_backbone_layers == 0:
            for layer in self.base_model.layers:
                layer.trainable = False
            return

        freeze_until = max(0, total_layers - trainable_backbone_layers)
        for layer in self.base_model.layers[:freeze_until]:
            layer.trainable = False
        for layer in self.base_model.layers[freeze_until:]:
            layer.trainable = True

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

    @staticmethod
    def _build_class_weight_map(y, min_weight=0.5, max_weight=5.0):
        y = np.asarray(y)
        classes, counts = np.unique(y, return_counts=True)
        total = float(np.sum(counts))
        n_classes = float(len(classes))

        class_weights = {}
        for class_id, class_count in zip(classes, counts):
            raw = total / (n_classes * float(class_count))
            class_weights[int(class_id)] = float(np.clip(raw, min_weight, max_weight))

        return class_weights

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

        class_weight_map = cls._build_class_weight_map(y_train, min_weight=0.3, max_weight=10.0)
        print(f"Class weights: {class_weight_map}")

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

        if epochs <= 1:
            warmup_epochs = epochs
        else:
            warmup_epochs = max(1, int(round(epochs * 0.3)))
            warmup_epochs = min(warmup_epochs, epochs - 1)

        model.set_fine_tune_layers(trainable_backbone_layers=0)
        model.compile(learning_rate=1e-3, loss="sparse_categorical_crossentropy")

        model.model.summary()

        reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-7, verbose=1)

        save_best = callbacks.ModelCheckpoint(
            model_filename, monitor="val_loss", save_best_only=True, mode="min", verbose=0
        )

        print(
            f"Starting training ResNetModel for {epochs} epochs (warmup: {warmup_epochs}, finetune: {epochs - warmup_epochs})..."
        )

        try:
            train_callbacks = [save_best, reduce_lr]
            if wandb_callback is not None:
                train_callbacks.append(wandb_callback)

            history_warmup = model.model.fit(
                X_train,
                y_train,
                epochs=warmup_epochs,
                batch_size=16,
                class_weight=class_weight_map,
                callbacks=train_callbacks,
                validation_data=(X_val, y_val),
            )

            model.set_fine_tune_layers(trainable_backbone_layers=60)
            model.compile(learning_rate=5e-5, loss="sparse_categorical_crossentropy")

            history_finetune = model.model.fit(
                X_train,
                y_train,
                initial_epoch=warmup_epochs,
                epochs=epochs,
                batch_size=8,
                class_weight=class_weight_map,
                callbacks=train_callbacks,
                validation_data=(X_val, y_val),
            )

            history_dict = {}
            for key in history_warmup.history:
                history_dict[key] = history_warmup.history[key] + history_finetune.history[key]

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
