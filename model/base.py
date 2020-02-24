# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from typing import Tuple, List, Dict, Any, Union, Optional
import pathlib
import tensorflow as tf
import numpy as np
import scipy.ndimage.morphology as morphology
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

    def load(
            self,
            path: Union[str, pathlib.Path]) -> None:
        """Load pre-trained model.

        Args:
            path (str or pathlib.Path): path to model file directory.

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


class TerminateOnValNaN(tf.keras.callbacks.Callback):
    """Callback that terminates training when a NaN validation loss is encountered."""

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        if loss is not None:
            if np.isnan(loss) or np.isinf(loss) or loss >= 100000:
                print('Batch %d: Invalid loss, terminating training' % (batch))
                self.model.stop_training = True
                raise Exception()

    def on_epoch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('val_loss')
        if loss is not None:
            if np.isnan(loss) or np.isinf(loss) or loss >= 100000:
                print('Batch %d: Invalid loss, terminating training' % (batch))
                self.model.stop_training = True
                raise Exception()


class KerasClassifierBase(KerasModelBase):
    """Keras image classification model base.

    Args:
        dataset (ImageClassifierDatasetBase): dataset object.
        epochs (int): number of training epochs.
        optimizer_name (str): optimizer class name.
        lr (float): initial learning rate.
        momentum (float): momentum value.
        clipnorm (float): clipnorm value
        lr_step_decay (bool): whether to use step learning rate decay.
        lr_step_span(float): learning rate decay span.
        decay (float): learning rate decay parameter.
        weighted_loss (List[float]): loss weight.
        restore_path (Union[str, pathlib.Path]): path to restore model file

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
            lr_step_span: float = 0.0,
            decay: float = 0.0,
            weighted_loss: Optional[List[float]] = None,
            restore_path: Union[str, pathlib.Path] = None,
            **kwargs: Any) -> None:
        """Intialize parameter and build model."""
        super(KerasClassifierBase, self).__init__(**kwargs)
        self.dataset = dataset
        self.epochs = epochs
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.momentum = momentum
        self.clipnorm = clipnorm
        self.lr_step_decay = lr_step_decay
        self.lr_step_span = lr_step_span
        self.decay = decay
        self.weighted_loss = weighted_loss
        self.restore_path = restore_path

    def _loss(self, label, pred):
        return tf.keras.losses.categorical_crossentropy(label, pred)

    @property
    def metrics(self):
        return [tf.keras.metrics.CategoricalAccuracy()]

    def setup(self) -> None:
        """Set optimizer to model."""
        config = {
                    'learning_rate': self.lr,
                    'momentum': self.momentum,
                    'clipnorm': self.clipnorm
                }

        if self.optimizer_name == 'adam':
            config = {
                        'learning_rate': self.lr,
                        'beta_1': self.momentum,
                        'clipnorm': self.clipnorm
                    }

        optimizer = tf.keras.optimizers.get({
            'class_name': self.optimizer_name,
            'config': config})

        if self.restore_path is not None:
            self.load(self.restore_path)

        self.model.compile(
            optimizer=optimizer,
            loss=self._loss,
            metrics=self.metrics)

    def train(self) -> Dict[str, List[Any]]:
        """Training model.

        Return:
            log (Dict[str, List[Any]]): training log.

        """
        callbacks: List = [
            TerminateOnValNaN(),
            tf.keras.callbacks.TensorBoard(write_graph=False, histogram_freq=1)
        ]
        if self.lr_step_decay:
            if self.lr_step_span != 0.0:
                callbacks.append(tf.keras.callbacks.LearningRateScheduler(
                        lambda e: self.lr * tf.math.pow(self.decay, (e // self.lr_step_span))))
            else:
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

    def load(
            self,
            path: Union[str, pathlib.Path]) -> None:
        """Load pre-trained model.

        Args:
            path (str or pathlib.Path): path to model file directory.

        """
        self.model.load_weights(str(path))


class KerasObjectDetectionBase(KerasClassifierBase):
    """Keras object detection model base."""

    pass


class KerasImageSegmentationBase(KerasClassifierBase):
    """Keras image segmentation model base.

    Args:
        generarized_dice_loss (Dict): parameters for generarized dice loss. must contains `alpha`.

    """

    def __init__(
            self,
            generarized_dice_loss: Dict = None,
            **kwargs: Any) -> None:
        """Intialize parameter and build model."""
        super(KerasImageSegmentationBase, self).__init__(**kwargs)
        self.generarized_dice_loss = generarized_dice_loss

    def setup(self) -> None:
        """Set optimizer to model."""
        optimizer = tf.keras.optimizers.get({
            'class_name': self.optimizer_name,
            'config': {
                'learning_rate': self.lr,
                'momentum': self.momentum,
                'clipnorm': self.clipnorm
                }})

        if self.restore_path is not None:
            self.load(self.restore_path)

        def weighted_logits(
                y_true: List,
                y_pred: List) -> float:
            """Return weighted loss."""
            return tf.nn.weighted_cross_entropy_with_logits(
                        logits=y_pred,
                        labels=y_true,
                        pos_weight=tf.constant(self.weighted_loss))

        def generarized_dice_loss(
                y_true: tf.Tensor,
                y_pred: tf.Tensor) -> float:
            """Return generarized dice loss.

            Reference:
                - Generarized Dice Loss: https://arxiv.org/abs/1707.03237
                - Boundary Loss: https://openreview.net/pdf?id=S1gTA5VggE

            """
            epsilon = tf.keras.backend.epsilon()
            w = 1 / (tf.square(tf.reduce_sum(y_true, axis=(1, 2))) + epsilon)
            intersection = tf.math.reduce_sum(y_true * y_pred, axis=(1, 2))
            union = tf.math.reduce_sum(y_true + y_pred, axis=(1, 2))
            gd_losses = 1 - 2 * (tf.math.reduce_sum(w * intersection, axis=-1) / (tf.math.reduce_sum(w * union, axis=-1) + epsilon))

            distance_negative = tf.cast(
                    tf.py_function(morphology.distance_transform_edt, [y_true[:, :, :, 0]], tf.double),
                    tf.keras.backend.floatx())
            distance_positive = tf.cast(
                    tf.py_function(morphology.distance_transform_edt, [y_true[:, :, :, 1]], tf.double),
                    tf.keras.backend.floatx())
            boundary_losses = (
                -w[:, 0] * tf.math.reduce_sum(distance_negative * y_true[:, :, :, 0] * y_pred[:, :, :, 0], axis=(1, 2)) +
                w[:, 1] * tf.math.reduce_sum(distance_positive * y_true[:, :, :, 1] * y_pred[:, :, :, 0], axis=(1, 2)))
            return tf.reduce_mean(gd_losses) + self.generarized_dice_loss['alpha'] * tf.reduce_mean(boundary_losses)

        if self.generarized_dice_loss is not None:
            loss = generarized_dice_loss
        elif self.weighted_loss is None:
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


class KerasLanguageModelBase(KerasClassifierBase):
    """Keras language model base."""

    def _loss(self, label, pred):
        mask = tf.math.logical_not(tf.math.equal(label, 0))
        loss = tf.keras.losses.sparse_categorical_crossentropy(label, pred)

        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask

        return loss

    @property
    def metrics(self):
        return [tf.keras.metrics.SparseCategoricalAccuracy()]
