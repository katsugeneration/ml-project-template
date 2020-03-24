# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from typing import Dict, List, Any
import pathlib
from matplotlib import pyplot as plt
import seaborn
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from runner.base import RunnerBase
from dataset.base import BinaryImageClassifierDataset
from dataset.mnist import MnistDataset
from dataset.cifar10 import Cifar10Dataset
from dataset.cifar100 import Cifar100Dataset
from dataset.mnist_from_raw import MnistFromRawDataset
from dataset.open_images import OpenImagesClassificationDataset
from model.base import KerasClassifierBase
from model.fcnn import FCNNClassifier
from model.cnn import ConvolutionalNet
from model.resnet import ResNet
from model.resnet101 import ResNet101
from model.efficientnet import EfficientNet
from model.yolo import YoloV2


class ImageRecognitionTrainer(RunnerBase[BinaryImageClassifierDataset, KerasClassifierBase]):
    """Image recognition task trainning runner."""

    def __init__(self):
        """Initilize parameters."""
        self.datasets = {
            'mnist': MnistDataset,
            'cifar10': Cifar10Dataset,
            'cifar100': Cifar100Dataset,
            'mnistraw': MnistFromRawDataset,
            'openimages': OpenImagesClassificationDataset,
        }

        self.models = {
            'fcnn': FCNNClassifier,
            'cnn': ConvolutionalNet,
            'resnet': ResNet,
            'resnet101': ResNet101,
            'efficientnet': EfficientNet,
            'yolo': YoloV2,
        }

    def _run(
            self,
            dataset: BinaryImageClassifierDataset,
            model: KerasClassifierBase,
            log_path: pathlib.Path) -> Dict[str, List[Any]]:
        """Run task.

        Args:
            dataset (BinaryImageClassifierDataset): dataset object.
            model (KerasClassifierBase): model object.
            log_path (pathlib.Path): log path object.

        Return:
            history (Dict[str, List[Any]]): task running history.

        """
        # run learning
        history = model.train()
        model.save(log_path.joinpath('model.h5'))

        # save results
        x_test, y_pred, y_test = model.inference()
        y_test = np.argmax(y_test, axis=1)
        y_pred = np.argmax(y_pred, axis=1)

        labels = list(range(10))
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        df_cm = pd.DataFrame(cm, index=labels, columns=labels)
        fig = plt.figure(figsize=(12.8, 7.2))
        seaborn.heatmap(df_cm, cmap=plt.cm.Blues, annot=True)
        fig.savefig(str(log_path.joinpath('confusion_matrix.png')))

        return history
