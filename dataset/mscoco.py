# Copyright 2020 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from typing import Tuple, List
import json
import pathlib
import zipfile
from collections import defaultdict
import concurrent.futures
import requests
import tqdm
from PIL import Image
import tensorflow as tf
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

    category_nums = 80

    image_feature_description = {
        'height': tf.io.FixedLenFeature([1], tf.int64),
        'width': tf.io.FixedLenFeature([1], tf.int64),
        'image': tf.io.FixedLenFeature([], tf.string),
    }

    label_feature_description = {
        'label': tf.io.FixedLenSequenceFeature([5], tf.float32),
    }

    def __init__(
            self,
            train_image_directory: pathlib.Path,
            test_image_directory: pathlib.Path,
            adjusted_shape: Tuple[int, int] = (416, 416),
            **kwargs) -> None:
        """Load and preprocessing coco object detection data.

        Args:
            adjusted_shape (Tuple): image adjusted size.

        """
        super(MSCococDatectionDataset, self).__init__(**kwargs)
        self.input_shape = (adjusted_shape[0], adjusted_shape[1], 3)

        self.x_train = list(train_image_directory.glob('*.*'))
        self.x_test = list(test_image_directory.glob('*.*'))


def make_example(args) -> bytes:
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
                        tf.train.Feature(float_list=tf.train.FloatList(value=l))
                        for l in label])
            })).SerializeToString()
    except Exception:
        return b""


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


def convert_tfrecord(
        artifact_directory: pathlib.Path,
        before_artifact_directory: pathlib.Path,
        split_num: int = 1) -> None:
    """Convert tfrecord format.

    Args:
        artifact_directory (Path): file save path.
        before_artifact_directory (Path): non use.

    """
    BBOX_KEY = 'bbox'
    CATEGORY_KEY = 'category_id'
    IMAGE_KEY = 'image_id'

    train_image_directory = before_artifact_directory.joinpath('train2014')
    train_label_path = before_artifact_directory.joinpath('annotations/instances_train2014.json')
    test_image_directory = before_artifact_directory.joinpath('val2014')
    test_label_path = before_artifact_directory.joinpath('annotations/instances_val2014.json')
    train_record_path = artifact_directory.joinpath('train')
    train_record_path.mkdir(parents=True, exist_ok=True)
    test_record_path = artifact_directory.joinpath('test')
    test_record_path.mkdir(parents=True, exist_ok=True)

    def preprocess(data, image_directory):
        _images = dict()
        _boxes = defaultdict(list)

        for image in data[IMAGES_KEY]:
            image_path = image_directory.joinpath(image[FILE_KEY])
            if image_path.exists():
                _images[image[ID_KEY]] = image_path
        for annotation in data[ANNOTATION_KEY]:
            left_x, left_y, w, h = annotation[BBOX_KEY]
            center_x = left_x + w / 2.
            center_y = left_y + h / 2.
            _boxes[annotation[IMAGE_KEY]].append(
                    [center_x, center_y, w, h, annotation[CATEGORY_KEY]])
        return _images, _boxes

    def _save(image_directory, label_path, tfrecord_path):
        with label_path.open() as f:
            data = json.load(f)
        images, boxes = preprocess(data, image_directory)
        keys = set(images.keys()) & set(boxes.keys())

        def generator():
            with concurrent.futures.ProcessPoolExecutor(5) as executor:
                for serialized_string in executor.map(make_example, [(images[k], boxes[k]) for k in keys]):
                    if len(serialized_string) != 0:
                        yield serialized_string

        def write(key, dataset):
            filename = tf.strings.join([str(tfrecord_path), '/data', tf.strings.as_string(key), '.tfrecord'])
            writer = tf.data.experimental.TFRecordWriter(filename)
            writer.write(dataset.map(lambda _, x: x))
            return tf.data.Dataset.from_tensors(filename)

        def key(i, *args):
            return i // split_num

        serialized_dataset = tf.data.Dataset.from_generator(
            generator, output_types=tf.string, output_shapes=())
        dataset = (serialized_dataset
                   .enumerate()
                   .apply(
                        tf.data.experimental.group_by_window(
                            key, write, split_num
                        )))
        writer = tf.data.experimental.TFRecordWriter(str(tfrecord_path) + '/files.tfrecord')
        writer.write(dataset)

    _save(train_image_directory, train_label_path, train_record_path)
    _save(test_image_directory, test_label_path, test_record_path)
