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

    def inference(self) -> Tuple[List[Any], List[Any]]:
        """Inference model.

        Return:
            inference (List[Any]): inference result.
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


class KerasModelBase(tf.keras.Model, ModelBase):
    """Keras model class."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        tf.keras.Model.__init__(self, *args, **kwargs)


class KerasImageClassifierBase(KerasModelBase):
    """Keras image classification model base.

    Args:
        dataset (ImageClassifierDatasetBase): dataset object.
        epochs (int): number of training epochs.
        optimizer_name (str): optimizer class name.
        lr (float): initial learning rate.
        momentum (float): momentum value.
        clipnorm (float): clipnorm value
        lr_step_decay (bool): whether to use step learning rate decay.

    """

    def __init__(
            self,
            dataset: ImageClassifierDatasetBase,
            epochs: int = 5,
            optimizer_name: str = "sgd",
            lr: float = 0.1,
            momentum: float = 0.9,
            clipnorm: float = 1.0,
            lr_step_decay: bool = True,
            **kwargs: Any) -> None:
        """Intialize parameter and build model."""
        super(KerasImageClassifierBase, self).__init__(**kwargs)
        self.dataset = dataset
        self.epochs = epochs
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.momentum = momentum
        self.clipnorm = clipnorm
        self.lr_step_decay = lr_step_decay

    def setup(self) -> None:
        """Set optimizer to model."""
        optimizer = tf.keras.optimizers.get({
            'class_name': self.optimizer_name,
            'config': {
                'learning_rate': self.lr,
                'momentum': self.momentum,
                'clipnorm': self.clipnorm
                }})

        super(KerasImageClassifierBase, self).compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

    def train(self) -> Dict[str, List[Any]]:
        """Training model.

        Return:
            log (Dict[str, List[Any]]): training log.

        """
        callbacks: List = []
        if self.lr_step_decay:
            callbacks.append(tf.keras.callbacks.LearningRateScheduler(
                    lambda e: self.lr if e < int(self.epochs / 2) else self.lr / 10.0 if e < int(self.epochs * 3 / 4) else self.lr / 100.0))

        generator = self.dataset.training_data_generator()
        (x_test, y_test) = self.dataset.eval_data()
        history = self.fit_generator(
                        generator,
                        steps_per_epoch=self.dataset.steps_per_epoch,
                        validation_data=(x_test, y_test),
                        epochs=self.epochs,
                        callbacks=callbacks)
        return history.history

    def inference(self) -> Tuple[List[List[float]], List[Any]]:
        """Inference model.

        Return:
            predicts (List[List[float]]): inference result. shape is data size x category_nums.
            gt (List[Any]): ground truth data.

        """
        (x_test, y_test) = self.dataset.eval_data()
        predicts = super(KerasImageClassifierBase, self).predict(x_test)
        return predicts, y_test

    def save(
            self,
            path: Union[str, pathlib.Path]) -> None:
        """Save model.

        Args:
            path (str or pathlib.Path): path to model save directory.

        """
        self.save_weights(str(path))
