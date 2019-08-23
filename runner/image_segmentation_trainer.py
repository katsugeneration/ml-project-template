# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from typing import Dict, List, Any
import pathlib
import numpy as np
from runner.base import RunnerBase
from runner import utils
from dataset.base import DatasetBase, DirectoryImageSegmentationDataset
from model.base import ModelBase
from model.pspnet import PSPNet


class ImageSegmentationTrainer(RunnerBase):
    """Image segmentation task trainning runner."""

    def __init__(self) -> None:
        """Initilize parameters."""
        self.datasets = {
            'directory': DirectoryImageSegmentationDataset,
        }

        self.models = {
            'pspnet': PSPNet,
        }

    def _run(
            self,
            dataset: DatasetBase,
            model: ModelBase,
            log_path: pathlib.Path) -> Dict[str, List[Any]]:
        """Run task.

        Args:
            dataset (DatasetBase): dataset object.
            model (ModelBase): model object.
            log_path (pathlib.Path): log path object.

        Return:
            history (Dict[str, List[Any]]): task running history.

        """
        # run learning
        history = model.train()
        model.save(log_path.joinpath('model.h5'))

        # save results
        y_pred, y_test = model.inference()
        y_test = np.argmax(y_test, axis=-1).flatten()
        y_pred = np.argmax(y_pred, axis=-1).flatten()
        utils.save_confusion_matrix(
            y_pred,
            y_test,
            dataset.category_nums,
            log_path.joinpath('confusion_matrix.png'))
        class_accuracies, prec, class_prec, rec, f1, iou = utils.evaluate_segmentation(
                                                            y_pred, y_test, dataset.category_nums)

        # save to history
        history['val_prec'] = [prec]
        history['val_rec'] = [rec]
        history['val_f1'] = [f1]
        history['val_iou'] = [iou]
        for i, acc in enumerate(class_accuracies):
            history['val_acc_{:02d}'.format(i)] = [acc]
        for i, prec in enumerate(class_prec):
            history['val_prec_{:02d}'.format(i)] = [prec]

        return history
