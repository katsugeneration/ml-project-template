# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from typing import Dict, List, Any
import pathlib
from PIL import Image, ImageDraw
import numpy as np
from runner.base import RunnerBase
from dataset.base import ObjectDitectionDatasetBase
from dataset.mscoco import MSCococDatectionDataset
from model.base import KerasObjectDetectionBase
from model.yolo import YoloV2


class ObjectDetectionTrainer(RunnerBase[ObjectDitectionDatasetBase, KerasObjectDetectionBase]):
    """Object detection task trainning runner."""

    def __init__(self):
        """Initilize parameters."""
        self.datasets = {
            'mscoco': MSCococDatectionDataset,
        }

        self.models = {
            'yolo': YoloV2,
        }

    def _run(
            self,
            dataset: ObjectDitectionDatasetBase,
            model: KerasObjectDetectionBase,
            log_path: pathlib.Path) -> Dict[str, List[Any]]:
        """Run task.

        Args:
            dataset (ObjectDitectionDatasetBase): dataset object.
            model (KerasObjectDetectionBase): model object.
            log_path (pathlib.Path): log path object.

        Return:
            history (Dict[str, List[Any]]): task running history.

        """
        # run learning
        history = model.train()
        model.save(log_path.joinpath('model.h5'))

        # save results
        x_test, y_pred, y_test = model.inference()

        for i, (image, boxes) in enumerate(zip(x_test[:10], y_pred[:10])):
            image = Image.fromarray((image.numpy() * 255.0).astype(np.uint8))
            draw = ImageDraw.Draw(image)
            for box in boxes:
                x, y, w, h, category = box[:5]
                x *= image.width
                w *= image.width
                y *= image.height
                h *= image.height
                draw.rectangle(((x - w / 2, y - h / 2), (x + w / 2, y + h / 2)), outline='red', width=5)
            image.save(log_path.joinpath('test{:2d}.png'.format(i)), "PNG")

        return history
