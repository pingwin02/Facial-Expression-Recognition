from abc import ABC, abstractmethod


class BaseModel(ABC):

    @abstractmethod
    def __init__(self, input_shape, num_classes):
        pass

    @abstractmethod
    def compile(self, optimizer="adam", loss="sparse_categorical_crossentropy", metrics=None, learning_rate=2e-4):
        pass

    @abstractmethod
    def predict(self, images_np):
        pass

    @classmethod
    @abstractmethod
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
        pass

    @classmethod
    @abstractmethod
    def eval(cls, val_tuple, label_map=None, dataset_name=None, dataset_path=None, train_tuple=None, cache_label=None):
        pass

    @staticmethod
    @abstractmethod
    def _prepare_inputs(X):
        pass

    @staticmethod
    @abstractmethod
    def _build_class_weight_map(y, min_weight=0.5, max_weight=5.0):
        pass
