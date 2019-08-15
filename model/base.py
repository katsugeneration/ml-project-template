from typing import Tuple, List, Dict, Any, Union
import pathlib
import tensorflow as tf
from dataset.base import ImageClassifierDatasetBase


class ModelBase(object):
    """Learning model base."""

    def __init__(self) -> None:
        """Intialize parameter and build model."""
        pass

    def train(self) -> Dict[str, List[Any]]:
        """Training model.

        Return:
            log (Dict[str, List[Any]]): training log.

        """
        pass

    def predict(self) -> Tuple[List[Any], List[Any]]:
        """Predict model.

        Return:
            predicts (List[Any]): predict result.
            gt (List[Any]): ground truth data.

        """
        pass

    def save(
            self,
            path: Union[str, pathlib.Path]) -> None:
        """Save model.

        Args:
            path (str or pathlib.Path): path to model save directory.

        """
        pass


class KerasClassifierBase(ModelBase):
    """Keras classifier model base.

    Args:
        optimizer (str): optimizer class name.
        lr (float): initial learning rate.
        momentum (float): momentum value.
        clipnorm (float): clipnorm value

    """

    def __init__(
            self,
            optimizer: str = "sgd",
            lr: float = 0.1,
            momentum: float = 0.9,
            clipnorm: float = 1.0) -> None:
        """Initialize parameters."""
        self.optimizer = optimizer
        self.lr = lr
        self.momentum = momentum
        self.clipnorm = clipnorm
        self.model: tf.keras.Model = None

    def compile(self) -> None:
        """Set optimizer to model."""
        optimizer = tf.keras.optimizers.get(self.optimizer)
        optimizer._set_hyper("learning_rate", self.lr)
        optimizer._set_hyper("momentum", self.momentum)
        optimizer.clipnorm = self.clipnorm

        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])


class KerasImageClassifierBase(KerasClassifierBase):
    """Keras image classification model base.

    Args:
        dataset (ImageClassifierDatasetBase): dataset object.
        epochs (int): number of training epochs.

    """

    def __init__(
            self,
            dataset: ImageClassifierDatasetBase,
            epochs: int = 5,
            **kwargs: Any) -> None:
        """Intialize parameter and build model."""
        super(KerasImageClassifierBase, self).__init__(**kwargs)
        self.dataset = dataset
        self.epochs = epochs

    def train(self) -> Dict[str, List[Any]]:
        """Training model.

        Return:
            log (Dict[str, List[Any]]): training log.

        """
        generator = self.dataset.training_data_generator()
        (x_test, y_test) = self.dataset.eval_data()
        history = self.model.fit_generator(
                        generator,
                        steps_per_epoch=self.dataset.steps_per_epoch,
                        validation_data=(x_test, y_test),
                        epochs=self.epochs)
        return history.history

    def predict(self) -> Tuple[List[List[float]], List[Any]]:
        """Predict model.

        Return:
            predicts (List[List[float]]): predict result. shape is data size x category_nums.
            gt (List[Any]): ground truth data.

        """
        (x_test, y_test) = self.dataset.eval_data()
        predicts = self.model.predict(x_test)
        return predicts, y_test

    def save(
            self,
            path: Union[str, pathlib.Path]) -> None:
        """Save model.

        Args:
            path (str or pathlib.Path): path to model save directory.

        """
        self.model.save_weights(str(path))
