# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from typing import Tuple, List, Dict, Any, Union, Optional
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

    def inference(self) -> Tuple[List[Any], List[Any], List[Any]]:
        """Inference model.

        Return:
            target (List[Any]): inference target.
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


class KerasModelBase(ModelBase):
    """Keras model class."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if int(tf.__version__.split('.')[0]) < 2 and not tf.compat.v1.executing_eagerly():
            tf.compat.v1.enable_eager_execution()
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[0], True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
            except RuntimeError:
                pass

        self.model: tf.keras.Model


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
        decay (float): learning rate decay parameter.
        weighted_loss (List[float]): loss weight.

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
            decay: float = 0.0,
            weighted_loss: Optional[List[float]] = None,
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
        self.weighted_loss = weighted_loss

    def setup(self) -> None:
        """Set optimizer to model."""
        optimizer = tf.keras.optimizers.get({
            'class_name': self.optimizer_name,
            'config': {
                'learning_rate': self.lr,
                'momentum': self.momentum,
                'clipnorm': self.clipnorm
                }})

        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
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
        eval_generator = self.dataset.eval_data_generator()
        history = self.model.fit_generator(
                        generator,
                        steps_per_epoch=self.dataset.steps_per_epoch,
                        validation_data=eval_generator,
                        validation_steps=self.dataset.eval_steps_per_epoch,
                        epochs=self.epochs,
                        callbacks=callbacks)
        return history.history

    def inference(self) -> Tuple[List[Any], List[List[Any]], List[Any]]:
        """Inference model.

        Return:
            target (List[Any]): inference target.
            predicts (List[List[float]]): inference result. shape is data size x category_nums.
            gt (List[Any]): ground truth data.

        """
        (x_test, y_test) = self.dataset.eval_data()
        predicts = self.model.predict(x_test)
        return x_test, predicts, y_test

    def save(
            self,
            path: Union[str, pathlib.Path]) -> None:
        """Save model.

        Args:
            path (str or pathlib.Path): path to model save directory.

        """
        self.model.save_weights(str(path))


class KerasImageSegmentationBase(KerasImageClassifierBase):
    """Keras image segmentation model base."""

    def setup(self) -> None:
        """Set optimizer to model."""
        optimizer = tf.keras.optimizers.get({
            'class_name': self.optimizer_name,
            'config': {
                'learning_rate': self.lr,
                'momentum': self.momentum,
                'clipnorm': self.clipnorm
                }})

        def weighted_logits(
                y_true: List,
                y_pred: List) -> float:
            """Return weighted loss."""
            return tf.nn.weighted_cross_entropy_with_logits(
                        logits=y_pred,
                        labels=y_true,
                        pos_weight=tf.constant(self.weighted_loss))

        if self.weighted_loss is None:
            loss = tf.keras.losses.categorical_crossentropy
        else:
            loss = weighted_logits

        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=(['accuracy', tf.keras.metrics.MeanIoU(num_classes=self.dataset.category_nums)]))

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
        history = self.model.fit_generator(
                        generator,
                        steps_per_epoch=self.dataset.steps_per_epoch,
                        validation_data=(x_test, y_test),
                        epochs=self.epochs,
                        max_queue_size=100,
                        callbacks=callbacks)
        return history.history
