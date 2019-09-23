# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from typing import Dict, List, Any
import pathlib
import numpy as np
from PIL import Image
from runner.base import RunnerBase
from runner import utils
from dataset.base import DatasetBase, DirectoryImageSegmentationDataset
from model.base import ModelBase
from model.pspnet import PSPNet


class ImageSegmentationEvaluator(RunnerBase):
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
        x_test, y_pred, y_test = model.inference()
        nums = x_test.shape[-1] // 4
        pos = nums // 2
        images_path = log_path.joinpath('images')
        images_path.mkdir(exist_ok=True)
        for i, (x, pre, y) in enumerate(zip(x_test, y_pred, y_test)):
            pre = (1 - np.argmax(pre, axis=-1)) * 255
            y = (1 - np.argmax(y, axis=-1)) * 255
            Image.fromarray((x[:, :, pos*4:pos*4+3] * 255).astype(np.uint8)).save(images_path.joinpath('%05d_image.jpg' % i))
            Image.fromarray(pre.astype(np.uint8)).save(images_path.joinpath('%05d_predicts.jpg' % i))
            Image.fromarray(y.astype(np.uint8)).save(images_path.joinpath('%05d_gt.jpg' % i))

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
        history = {}
        history['val_prec'] = [prec]
        history['val_rec'] = [rec]
        history['val_f1'] = [f1]
        history['val_iou'] = [iou]
        for i, acc in enumerate(class_accuracies):
            history['val_acc_{:02d}'.format(i)] = [acc]
        for i, prec in enumerate(class_prec):
            history['val_prec_{:02d}'.format(i)] = [prec]

        return history
