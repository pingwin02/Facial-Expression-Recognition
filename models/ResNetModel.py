import numpy as np
import tensorflow as tf
from keras import applications, callbacks, layers, models, optimizers

from models._base import BaseModel
from models._data import ArrayBatchSequence, resolve_input_shape, should_stream_batches
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
    def _adapt_input_channels(X, expected_channels=3):
        if getattr(X, "ndim", None) is None or X.ndim < 3:
            return X

        current_channels = X.shape[-1]
        if current_channels is None or int(current_channels) <= int(expected_channels):
            return X

        return X[..., :expected_channels]

    @staticmethod
    def _prepare_inputs(X):
        X = ResNetModel._adapt_input_channels(X)
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

        class_weight_map = cls._build_class_weight_map(y_train, min_weight=0.3, max_weight=10.0)
        print(f"Class weights: {class_weight_map}")

        X_train_model = cls._adapt_input_channels(X_train)
        X_val_model = cls._adapt_input_channels(X_val)

        num_classes = len(np.unique(np.concatenate([y_train, y_val])))
        input_shape = resolve_input_shape(X_train_model)
        use_streaming_batches = should_stream_batches(X_train_model) or should_stream_batches(X_val_model)

        if use_streaming_batches:
            print("Using batch-streamed training inputs to avoid full RAM materialization.")
            X_train_fit = None
            X_val_fit = None
        else:
            X_train_fit = cls._prepare_inputs(X_train_model)
            X_val_fit = cls._prepare_inputs(X_val_model)

            if X_train_fit.ndim == 4:
                X_train_fit = np.expand_dims(X_train_fit, axis=1)
            if X_val_fit.ndim == 4:
                X_val_fit = np.expand_dims(X_val_fit, axis=1)

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
            warmup_callbacks = [save_best, reduce_lr]
            if wandb_callback is not None:
                warmup_callbacks.append(wandb_callback)

            if use_streaming_batches:
                warmup_train_data = ArrayBatchSequence(
                    X_train_model,
                    y_train,
                    batch_size=16,
                    shuffle=True,
                    class_weight_map=class_weight_map,
                )
                warmup_val_data = ArrayBatchSequence(
                    X_val_model,
                    y_val,
                    batch_size=16,
                    shuffle=False,
                )
            else:
                warmup_train_data = X_train_fit
                warmup_val_data = (X_val_fit, y_val)

            history_warmup = model.model.fit(
                warmup_train_data,
                epochs=warmup_epochs,
                batch_size=None if use_streaming_batches else 16,
                class_weight=None if use_streaming_batches else class_weight_map,
                callbacks=warmup_callbacks,
                validation_data=warmup_val_data,
            )

            if epochs > warmup_epochs:
                print(f"Warmup phase completed. Starting fine-tuning for remaining {epochs - warmup_epochs} epochs...")
                model.set_fine_tune_layers(trainable_backbone_layers=80)
                model.compile(learning_rate=5e-5, loss="sparse_categorical_crossentropy")

                finetune_callbacks = [reduce_lr, save_best]
                if wandb_callback is not None:
                    finetune_callbacks.append(wandb_callback)

                if use_streaming_batches:
                    finetune_train_data = ArrayBatchSequence(
                        X_train_model,
                        y_train,
                        batch_size=8,
                        shuffle=True,
                        class_weight_map=class_weight_map,
                    )
                    finetune_val_data = ArrayBatchSequence(
                        X_val_model,
                        y_val,
                        batch_size=8,
                        shuffle=False,
                    )
                else:
                    finetune_train_data = X_train_fit
                    finetune_val_data = (X_val_fit, y_val)

                history_finetune = model.model.fit(
                    finetune_train_data,
                    initial_epoch=warmup_epochs,
                    epochs=epochs,
                    batch_size=None if use_streaming_batches else 8,
                    class_weight=None if use_streaming_batches else class_weight_map,
                    callbacks=finetune_callbacks,
                    validation_data=finetune_val_data,
                )

                history_dict = history_warmup.history
                for key, values in history_finetune.history.items():
                    history_dict.setdefault(key, [])
                    history_dict[key].extend(values)
            else:
                history_dict = history_warmup.history

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
