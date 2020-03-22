# Copyright 2020 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from typing import Tuple, DefaultDict, List
import pathlib
from collections import defaultdict
import pandas as pd
from PIL import Image
import tensorflow as tf
from dataset.base import DirectoryImageClassifierDataset


class OpenImagesClassificationDataset(DirectoryImageClassifierDataset):
    """Open Images dataset loader.

    An image could be labled multiple classes.
    """

    category_nums = 19_958  # All imabe level lables nums

    image_feature_description = {
        'height': tf.io.FixedLenFeature([1], tf.int64),
        'width': tf.io.FixedLenFeature([1], tf.int64),
        'image': tf.io.FixedLenFeature([], tf.string),
    }

    label_feature_description = {
        'label': tf.io.FixedLenSequenceFeature([1], tf.int64),
    }

    def __init__(
            self,
            train_image_directory: pathlib.Path,
            test_image_directory: pathlib.Path,
            adjusted_shape: Tuple[int, int] = (512, 512),
            **kwargs) -> None:
        """Load and preprocessing coco object detection data.

        Args:
            adjusted_shape (Tuple): image adjusted size.

        """
        super(OpenImagesClassificationDataset, self).__init__(**kwargs)
        self.input_shape = (adjusted_shape[0], adjusted_shape[1], 3)

        self.x_train = list(train_image_directory.glob('*.jpg'))
        self.x_test = list(test_image_directory.glob('*.jpg'))

    def _preprocess(
            self,
            image: tf.Tensor,
            label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Preprocess image and label.

        Args:
            image (array): image data array.
            label (array): label data array.

        Return:
            image (array): processed image data array.
            label (array): processed label data array.

        """
        image = tf.image.resize(image, self.input_shape[:2], tf.image.ResizeMethod.BILINEAR)
        image = image / 255.0
        label = tf.reduce_sum(tf.one_hot(tf.reshape(label, (-1, )), self.category_nums, axis=-1), axis=0)
        return image, label

    @classmethod
    def make_example(cls, args) -> bytes:
        """Convert image and label prot string.

        Args:
            image_path (str): image path string.
            label (List[float]): bbox label array.

        Return:
            proto (bytes): serialized protocol buffer string.

        """
        image_path, label = args
        try:
            image = Image.open(image_path).convert('RGB')
            image_string = image.tobytes()
            return tf.train.SequenceExample(context=tf.train.Features(feature={
                    'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[image.height])),
                    'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[image.width])),
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_string])),
                }),
                feature_lists=tf.train.FeatureLists(feature_list={
                        'label': tf.train.FeatureList(feature=[
                            tf.train.Feature(int64_list=tf.train.Int64List(value=[l]))
                            for l in label])
                })).SerializeToString()
        except Exception:
            return b""

    @classmethod
    def convert_tfrecord(
            cls,
            artifact_directory: pathlib.Path,
            before_artifact_directory: pathlib.Path,
            split_num: int = 1) -> None:
        """Convert tfrecord format.

        Args:
            artifact_directory (Path): file save path.
            before_artifact_directory (Path): non use.
            split_num (int): images nums per one tfrecord file.

        """
        ID_KEY = 'ImageID'
        LABEL_KEY = 'LabelName'
        CONFIDENCE_KEY = 'Confidence'

        train_image_directory = before_artifact_directory.joinpath('train')
        train_label_path = before_artifact_directory.joinpath('train-annotations-human-imagelabels-boxable.csv')
        test_image_directory = before_artifact_directory.joinpath('validation')
        test_label_path = before_artifact_directory.joinpath('validation-annotations-human-imagelabels-boxable.csv')
        class_name_path = before_artifact_directory.joinpath('oidv6-class-descriptions.csv')

        train_record_path = artifact_directory.joinpath('train')
        train_record_path.mkdir(parents=True, exist_ok=True)
        test_record_path = artifact_directory.joinpath('test')
        test_record_path.mkdir(parents=True, exist_ok=True)

        class_names = {k: v for v, k in enumerate(pd.read_csv(class_name_path, header=0)[LABEL_KEY])}

        def _preprocess(label_path: pathlib.Path) -> DefaultDict:
            """Select only positive class and image id pairs."""
            _labels: DefaultDict[str, List] = defaultdict(list)
            labels = pd.read_csv(label_path, header=0)
            labels = labels[labels[CONFIDENCE_KEY] == 1]

            for _, _id, l in labels[[ID_KEY, LABEL_KEY]].itertuples():
                _labels[_id].append(class_names[l])
            return _labels

        def _save(image_directory: pathlib.Path, label_path, tfrecord_path):
            labels = _preprocess(label_path)
            images = {im.stem: str(im) for im in image_directory.glob("*.jpg")}
            keys = set(images) & set(labels)
            cls._convert(images, labels, keys, tfrecord_path, split_num)

        _save(train_image_directory, train_label_path, train_record_path)
        _save(test_image_directory, test_label_path, test_record_path)
