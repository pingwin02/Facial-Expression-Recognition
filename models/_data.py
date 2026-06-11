import math

import numpy as np
from keras.utils import Sequence

STREAMING_THRESHOLD_BYTES = 1024 ** 3


def should_stream_batches(X):
    return isinstance(X, np.memmap) or int(getattr(X, "nbytes", 0)) >= STREAMING_THRESHOLD_BYTES


def resolve_input_shape(X):
    if getattr(X, "ndim", None) == 4:
        return (1, *X.shape[1:])
    return tuple(X.shape[1:])


class ArrayBatchSequence(Sequence):
    def __init__(self, X, y, batch_size, shuffle=False, class_weight_map=None):
        super().__init__()
        self.X = X
        self.y = np.asarray(y)
        self.batch_size = max(1, int(batch_size))
        self.shuffle = bool(shuffle)
        self.class_weight_map = class_weight_map or None
        self.indices = np.arange(len(self.y), dtype=np.int64)
        self.on_epoch_end()

    def __len__(self):
        return int(math.ceil(len(self.indices) / float(self.batch_size)))

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min(len(self.indices), start + self.batch_size)
        batch_indices = self.indices[start:end]

        batch_X = np.asarray(self.X[batch_indices], dtype=np.float32)
        if batch_X.size > 0 and np.max(batch_X) > 1.0:
            batch_X = batch_X / 255.0
        if batch_X.ndim == 4:
            batch_X = np.expand_dims(batch_X, axis=1)

        batch_y = self.y[batch_indices]

        if self.class_weight_map is None:
            return batch_X, batch_y

        batch_weights = np.asarray(
            [self.class_weight_map.get(int(label), 1.0) for label in batch_y],
            dtype=np.float32,
        )
        return batch_X, batch_y, batch_weights

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
