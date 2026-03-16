import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, applications

from utils.eval import evaluate_model_on_data
from utils.model_io import find_and_load_model
from utils.plotting import plot_metrics
from utils.wandb_utils import init_wandb_run, finish_wandb_run


class TransferModel:
    def __init__(self, input_shape=(8, 48, 48, 2), num_classes=7):
        inputs = layers.Input(shape=input_shape)

        data_augmentation = models.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomContrast(0.12),
                layers.RandomRotation(0.06),
                layers.RandomZoom(height_factor=(-0.08, 0.12), width_factor=(-0.08, 0.12)),
            ],
            name="frame_aug",
        )

        x = layers.TimeDistributed(data_augmentation, name="video_aug")(inputs)

        x = layers.TimeDistributed(layers.Conv2D(24, (1, 1), padding="same", use_bias=False))(x)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        x = layers.TimeDistributed(layers.Activation("relu"))(x)

        x = layers.TimeDistributed(layers.SeparableConv2D(24, (3, 3), padding="same", use_bias=False))(x)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        x = layers.TimeDistributed(layers.Activation("relu"))(x)

        x = layers.TimeDistributed(layers.Conv2D(3, (1, 1), padding="same", use_bias=False))(x)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        x = layers.TimeDistributed(layers.Activation("relu"))(x)

        x = layers.TimeDistributed(layers.Resizing(128, 128, interpolation="bilinear"))(x)

        x = layers.TimeDistributed(layers.Rescaling(scale=2.0, offset=-1.0))(x)

        self.base_model = applications.EfficientNetV2B2(
            input_shape=(128, 128, 3), include_top=False, weights="imagenet"
        )

        x = layers.TimeDistributed(self.base_model)(x)
        x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        x = layers.TimeDistributed(layers.Dropout(0.35))(x)

        timesteps = int(input_shape[0]) if isinstance(input_shape, tuple) and len(input_shape) > 0 else None
        if timesteps == 1:
            x = layers.Flatten(name="single_frame_pool")(x)
        else:
            x = layers.Bidirectional(layers.GRU(192, return_sequences=True, dropout=0.25, recurrent_dropout=0.0))(x)
            x = layers.LayerNormalization()(x)
            x = layers.Bidirectional(layers.GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.0))(x)

            attn = layers.Dense(1, activation="tanh", name="temporal_attention_logits")(x)
            attn = layers.Softmax(axis=1, name="temporal_attention_weights")(attn)
            x = layers.Dot(axes=1, name="temporal_attention_pool")([attn, x])
            x = layers.Flatten()(x)

        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)

        x = layers.Dense(512)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("gelu")(x)
        x = layers.Dropout(0.4)(x)

        x = layers.Dense(256)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("gelu")(x)
        x = layers.Dropout(0.3)(x)

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

    def compile(self, optimizer="adam", loss="sparse_categorical_crossentropy", metrics=None, learning_rate=2e-4):
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
    def _oversample_minority_classes(X, y, min_target_ratio=0.35, max_multiplier=8, seed=42):
        y = np.asarray(y)
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) <= 1:
            return X, y

        max_count = int(np.max(counts))
        rng = np.random.default_rng(seed)

        extra_indices = []
        for class_id, class_count in zip(classes, counts):
            target_from_ratio = int(max_count * min_target_ratio)
            target_from_multiplier = int(class_count * max_multiplier)
            target_count = max(class_count, min(target_from_ratio, target_from_multiplier))

            if target_count <= class_count:
                continue

            class_indices = np.where(y == class_id)[0]
            add_count = int(target_count - class_count)
            sampled = rng.choice(class_indices, size=add_count, replace=True)
            extra_indices.append(sampled)

        if not extra_indices:
            return X, y

        extra_indices = np.concatenate(extra_indices)
        X_balanced = np.concatenate([X, X[extra_indices]], axis=0)
        y_balanced = np.concatenate([y, y[extra_indices]], axis=0)

        shuffle_idx = rng.permutation(len(y_balanced))
        return X_balanced[shuffle_idx], y_balanced[shuffle_idx]

    @staticmethod
    def _build_class_weight_map(y, min_weight=0.8, max_weight=2.5):
        y = np.asarray(y)
        classes, counts = np.unique(y, return_counts=True)
        total = float(np.sum(counts))
        n_classes = float(len(classes))

        class_weights = {}
        for class_id, class_count in zip(classes, counts):
            raw = total / (n_classes * float(class_count))
            class_weights[int(class_id)] = float(np.clip(raw, min_weight, max_weight))

        return class_weights

    @staticmethod
    def _build_sample_weights(y, class_weights):
        y = np.asarray(y)
        sample_weights = np.array([class_weights.get(int(lbl), 1.0) for lbl in y], dtype=np.float32)

        return sample_weights

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
    ):
        wandb_run = None
        wandb_callback = None

        X_train = cls._prepare_inputs(X_train)
        X_val = cls._prepare_inputs(X_val)

        if X_train.ndim == 4:
            X_train = np.expand_dims(X_train, axis=1)
        if X_val.ndim == 4:
            X_val = np.expand_dims(X_val, axis=1)

        X_train, y_train = cls._oversample_minority_classes(X_train, y_train, min_target_ratio=0.45, max_multiplier=6)

        class_weight_map = cls._build_class_weight_map(y_train)
        sample_weights = cls._build_sample_weights(y_train, class_weight_map)
        print(f"Class weights: {class_weight_map}")

        num_classes = len(np.unique(np.concatenate([y_train, y_val])))
        input_shape = X_train.shape[1:]

        model = cls(input_shape=input_shape, num_classes=num_classes)
        warmup_epochs = min(max(8, epochs // 8), 20)
        warmup_epochs = min(warmup_epochs, max(1, epochs))

        wandb_run, wandb_callback = init_wandb_run(
            model_name=cls.__name__,
            dataset_name=dataset_name,
            epochs=epochs,
            output_dir=output_dir,
            extra_config={
                "num_classes": int(num_classes),
                "input_shape": tuple(input_shape),
                "warmup_epochs": int(warmup_epochs),
            },
        )

        model.set_fine_tune_layers(trainable_backbone_layers=0)
        model.compile(learning_rate=3e-4, loss="sparse_categorical_crossentropy")

        model.model.summary()

        reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-7, verbose=1)

        save_best = callbacks.ModelCheckpoint(
            model_filename, monitor="val_loss", save_best_only=True, mode="min", verbose=0
        )

        print(f"Starting training TransferModel for {epochs} epochs...")

        try:
            warmup_callbacks = [save_best, reduce_lr]
            if wandb_callback is not None:
                warmup_callbacks.append(wandb_callback)

            history_warmup = model.model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=warmup_epochs,
                batch_size=8,
                sample_weight=sample_weights,
                callbacks=warmup_callbacks,
            )

            if epochs > warmup_epochs:
                model.set_fine_tune_layers(trainable_backbone_layers=120)
                model.compile(learning_rate=4e-5, loss="sparse_categorical_crossentropy")

                finetune_callbacks = [reduce_lr, save_best]
                if wandb_callback is not None:
                    finetune_callbacks.append(wandb_callback)

                history_finetune = model.model.fit(
                    X_train,
                    y_train,
                    validation_data=(X_val, y_val),
                    initial_epoch=warmup_epochs,
                    epochs=epochs,
                    batch_size=8,
                    callbacks=finetune_callbacks,
                    sample_weight=sample_weights,
                )

                history_dict = history_warmup.history
                for key, values in history_finetune.history.items():
                    history_dict.setdefault(key, [])
                    history_dict[key].extend(values)
            else:
                history_dict = history_warmup.history

            model.model.load_weights(model_filename)
            model.model.save(model_filename)
            plot_metrics(
                history_dict,
                output_dir,
                model_name=cls.__name__,
                training_debugs=train_debugs,
                validation_debugs=val_debugs,
                dataset_name=dataset_name,
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
    def eval(cls, val_tuple, output_dir, label_map=None, dataset_name=None, dataset_path=None, train_tuple=None):
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
            dataset_path=dataset_path,
            train_tuple=train_tuple,
        )
