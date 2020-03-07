# Copyright 2020 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from typing import Tuple
import json
import pathlib
import zipfile
from collections import defaultdict
import requests
import tqdm
from dataset.base import DirectoryObjectDitectionDataset

BASE_URL = 'http://images.cocodataset.org/zips/'
BASE_ANNOTATION_URL = 'http://images.cocodataset.org/annotations/'
TRAIN_FILE = 'train2014.zip'
EVAL_FILE = 'val2014.zip'
TEST_FILE = 'test2014.zip'
TRAIN_ANN_FILE = 'annotations_trainval2014.zip'

TRAIN_KEY = 'train2014'
EVAL_KEY = 'val2014'
ANNOTATION_KEY = 'annotations'
IMAGES_KEY = 'images'
ID_KEY = 'id'
FILE_KEY = 'file_name'


class MSCococDatectionDataset(DirectoryObjectDitectionDataset):
    """MSCOCO dataset loader."""

    BBOX_KEY = 'bbox'
    CATEGORY_KEY = 'category_id'
    IMAGE_KEY = 'image_id'

    category_nums = 80

    def __init__(
            self,
            adjusted_shape: Tuple[int, int] = (416, 416),
            **kwargs) -> None:
        """Load and preprocessing coco object detection data.

        Args:
            adjusted_shape (Tuple): image adjusted size.

        """
        super(MSCococDatectionDataset, self).__init__(**kwargs)

        self.input_shape = (adjusted_shape[0], adjusted_shape[1], 3)

        def preprocess(data, image_directory):
            _images = dict()
            _boxes = defaultdict(list)

            for image in data[IMAGES_KEY]:
                image_path = image_directory.joinpath(image[FILE_KEY])
                if image_path.exists():
                    _images[image[ID_KEY]] = image_path
            for annotation in data[ANNOTATION_KEY]:
                left_x, left_y, w, h = annotation[self.BBOX_KEY]
                center_x = left_x + w / 2.
                center_y = left_y + h / 2.
                _boxes[annotation[self.IMAGE_KEY]].append(
                     [center_x, center_y, w, h, annotation[self.CATEGORY_KEY]])
            return _images, _boxes

        with self.train_label_path.open() as f:
            data = json.load(f)
        train_images, train_boxes = preprocess(data, self.train_image_directory)
        train_keys = set(train_images.keys()) & set(train_boxes.keys())
        self.x_train = [train_images[k] for k in train_keys]
        self.y_train = [train_boxes[k] for k in train_keys]

        with self.test_label_path.open() as f:
            data = json.load(f)
        test_images, test_boxes = preprocess(data, self.test_image_directory)
        test_keys = set(test_images.keys()) & set(test_boxes.keys())
        self.x_test = [test_images[k] for k in test_keys]
        self.y_test = [test_boxes[k] for k in test_keys]


def download(
        artifact_directory: pathlib.Path,
        before_artifact_directory: pathlib.Path = None) -> None:
    """Download MSCOCO image and annotation data from cocodataset.org.

    Args:
        artifact_directory (Path): file save path.
        before_artifact_directory (Path): non use.

    """
    save_path = artifact_directory
    save_path.mkdir(parents=True, exist_ok=True)

    for f in [TRAIN_FILE, EVAL_FILE, TEST_FILE]:
        file_path = save_path.joinpath(f)
        res = requests.get(BASE_URL + f, stream=True)
        size = int(res.headers['Content-Length'])
        pbar = tqdm.tqdm(desc=str(f), total=size, unit='B', unit_scale=True)
        with file_path.open('wb') as w:
            for buf in res.iter_content(chunk_size=1024**2):
                w.write(buf)
                pbar.update(len(buf))
        res.close()

        with zipfile.ZipFile(file_path) as zip:
            zip.extractall(artifact_directory)
        file_path.unlink()

    for f in [TRAIN_ANN_FILE]:
        file_path = save_path.joinpath(f)
        res = requests.get(BASE_ANNOTATION_URL + f, stream=True)
        size = int(res.headers['Content-Length'])
        pbar = tqdm.tqdm(desc=str(f), total=size, unit='B', unit_scale=True)
        with file_path.open('wb') as w:
            for buf in res.iter_content(chunk_size=1024**2):
                w.write(buf)
                pbar.update(len(buf))
        res.close()

        with zipfile.ZipFile(file_path) as zip:
            zip.extractall(artifact_directory)
        file_path.unlink()
