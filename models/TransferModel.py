import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, applications

from utils.eval import evaluate_model_on_data
from utils.model_io import find_and_load_model
from utils.plotting import plot_metrics


class TransferModel:
    def __init__(self, input_shape=(8, 48, 48, 2), num_classes=7):
        inputs = layers.Input(shape=input_shape)

        data_augmentation = models.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomContrast(0.12),
            ],
            name="frame_aug",
        )

        x = layers.TimeDistributed(data_augmentation, name="video_aug")(inputs)

        x = layers.TimeDistributed(layers.Conv2D(16, (1, 1), padding="same", use_bias=False))(x)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        x = layers.TimeDistributed(layers.Activation("relu"))(x)

        x = layers.TimeDistributed(layers.Conv2D(3, (3, 3), padding="same", use_bias=False))(x)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        x = layers.TimeDistributed(layers.Activation("relu"))(x)

        x = layers.TimeDistributed(layers.Resizing(96, 96, interpolation="bilinear"))(x)

        x = layers.TimeDistributed(layers.Rescaling(scale=2.0, offset=-1.0))(x)

        self.base_model = applications.EfficientNetV2B0(input_shape=(96, 96, 3), include_top=False, weights="imagenet")

        x = layers.TimeDistributed(self.base_model)(x)
        x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        x = layers.TimeDistributed(layers.Dropout(0.25))(x)

        timesteps = int(input_shape[0]) if isinstance(input_shape, tuple) and len(input_shape) > 0 else None
        if timesteps == 1:
            x = layers.Flatten(name="single_frame_pool")(x)
        else:
            x = layers.Bidirectional(layers.GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.0))(x)

            attn = layers.Dense(1, activation="tanh", name="temporal_attention_logits")(x)
            attn = layers.Softmax(axis=1, name="temporal_attention_weights")(attn)
            x = layers.Dot(axes=1, name="temporal_attention_pool")([attn, x])
            x = layers.Flatten()(x)

        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)

        x = layers.Dense(256)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
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
    def _build_class_weight_map(y, min_weight=0.5, max_weight=8.0):
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
    def train(cls, X_train, y_train, X_val, y_val, output_dir, model_filename, epochs, label_map):
        X_train = cls._prepare_inputs(X_train)
        X_val = cls._prepare_inputs(X_val)

        if X_train.ndim == 4:
            X_train = np.expand_dims(X_train, axis=1)
        if X_val.ndim == 4:
            X_val = np.expand_dims(X_val, axis=1)

        y_train_original = np.asarray(y_train)

        if X_train.ndim == 5:
            X_train, y_train = cls._oversample_minority_classes(
                X_train, y_train, min_target_ratio=0.2, max_multiplier=4
            )
        else:
            X_train, y_train = cls._oversample_minority_classes(X_train, y_train)

        class_weight_map = cls._build_class_weight_map(y_train_original)
        sample_weights = cls._build_sample_weights(y_train, class_weight_map)
        val_sample_weights = cls._build_sample_weights(y_val, class_weight_map)
        print(f"Class weights: {class_weight_map}")

        num_classes = len(np.unique(np.concatenate([y_train, y_val])))
        input_shape = X_train.shape[1:]

        model = cls(input_shape=input_shape, num_classes=num_classes)
        warmup_epochs = min(max(5, epochs // 10), 15)
        warmup_epochs = min(warmup_epochs, max(1, epochs))

        model.set_fine_tune_layers(trainable_backbone_layers=0)
        model.compile(learning_rate=5e-4, loss="sparse_categorical_crossentropy")

        model.model.summary()

        reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-7, verbose=1)

        save_best = callbacks.ModelCheckpoint(
            model_filename, monitor="val_loss", save_best_only=True, mode="min", verbose=0
        )

        print(f"Starting training TransferModel for {epochs} epochs...")

        history_warmup = model.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val, val_sample_weights),
            epochs=warmup_epochs,
            batch_size=8,
            sample_weight=sample_weights,
            callbacks=[save_best],
        )

        if epochs > warmup_epochs:
            model.set_fine_tune_layers(trainable_backbone_layers=60)
            model.compile(learning_rate=8e-5, loss="sparse_categorical_crossentropy")

            history_finetune = model.model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val, val_sample_weights),
                initial_epoch=warmup_epochs,
                epochs=epochs,
                batch_size=8,
                callbacks=[reduce_lr, save_best],
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
        plot_metrics(history_dict, output_dir, model_name=cls.__name__)

        return history_dict

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
