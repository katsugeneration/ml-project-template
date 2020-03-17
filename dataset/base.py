# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from typing import Tuple, Any, Generator, Union, List, Optional
import pathlib
import random
import multiprocessing
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
from dataset.mixup import MixupGenerator


class DatasetBase(object):
    """Dataset loader base class."""

    def __init__(self) -> None:
        """Load data and setup preprocessing."""
        pass

    def training_data(self) -> Tuple[np.array, np.array]:
        """Return training dataset.

        Return:
            dataset (Tuple[np.array, np.array]): training dataset pair

        """
        pass

    def training_data_generator(self) -> Union[tf.keras.utils.Sequence, Generator]:
        """Return training dataset iterator.

        Return:
            dataset (Union[tf.keras.utils.Sequence, Generator]): dataset iterator

        """
        pass

    def eval_data(self) -> Tuple[np.array, np.array]:
        """Return evaluation dataset.

        Return:
            dataset (Tuple[np.array, np.array]): evaluation dataset pair

        """
        pass

    def eval_data_generator(self) -> Union[tf.keras.utils.Sequence, Generator]:
        """Return evaluation dataset.

        Return:
            dataset (Union[tf.keras.utils.Sequence, Generator]): dataset generator

        """
        pass


class ImageClassifierDatasetBase(DatasetBase):
    """Image classification dataset loader base class.

    Args:
        batch_size (int): training batch size.
        use_mixup (bool): whether to use mixup augmentation.

    """

    def __init__(
            self,
            batch_size: int = 32,
            use_mixup: bool = False,
            **kwargs: Any
            ) -> None:
        """Initialize data generator."""
        self.batch_size = batch_size
        self.use_mixup = use_mixup
        self.train_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(**kwargs)
        self.eval_data_gen = tf.keras.preprocessing.image.ImageDataGenerator()
        self.input_shape: Tuple[int, int, int]
        self.category_nums: int
        self.steps_per_epoch: int
        self.eval_steps_per_epoch: int


class BinaryImageClassifierDataset(ImageClassifierDatasetBase):
    """Memory loaded classifier dataset."""

    def __init__(self, **kwargs: Any) -> None:
        super(BinaryImageClassifierDataset, self).__init__(**kwargs)
        self.x_train: np.array
        self.x_test: np.array
        self.y_train: np.array
        self.y_test: np.array

    def training_data(self) -> Tuple[np.array, np.array]:
        """Return training dataset.

        Return:
            dataset (Tuple[np.array, np.array]): training dataset pair

        """
        y_train = tf.keras.utils.to_categorical(self.y_train)
        return (self.train_data_gen.random_transform(self.x_train), y_train)

    def training_data_generator(self) -> Union[tf.keras.utils.Sequence, Generator]:
        """Return training dataset.

        Return:
            dataset (Union[tf.keras.utils.Sequence, Generator]): dataset generator

        """
        y_train = tf.keras.utils.to_categorical(self.y_train)
        self.steps_per_epoch = len(self.x_train) // self.batch_size
        if self.use_mixup:
            return MixupGenerator(datagen=self.train_data_gen).flow(self.x_train, y_train, batch_size=self.batch_size)
        else:
            return self.train_data_gen.flow(self.x_train, y=y_train, batch_size=self.batch_size)

    def eval_data(self) -> Tuple[np.array, np.array]:
        """Return evaluation dataset.

        Return:
            dataset (Tuple[np.array, np.array]): evaluation dataset pair

        """
        y_test = tf.keras.utils.to_categorical(self.y_test)
        return (self.x_test, y_test)

    def eval_data_generator(self) -> Union[tf.keras.utils.Sequence, Generator]:
        """Return evaluation dataset.

        Return:
            dataset (Union[tf.keras.utils.Sequence, Generator]): dataset generator

        """
        y_test = tf.keras.utils.to_categorical(self.y_test)
        self.eval_steps_per_epoch = len(self.x_test) // self.batch_size
        return self.eval_data_gen.flow(self.x_test, y=y_test, batch_size=self.batch_size)


class ObjectDitectionDatasetBase(ImageClassifierDatasetBase):
    """Object detection dataset loader base class.

    Note:
        label is Ground truth boxes tensor with shape (num_true_boxes, 5)
        containing box relative x_center, y_center, width, height, and class.

    """

    def __init__(
            self,
            **kwargs: Any) -> None:
        """Initilize params."""
        super(ObjectDitectionDatasetBase, self).__init__(**kwargs)
        self.max_boxes: int


class DirectoryObjectDitectionDataset(ObjectDitectionDatasetBase):
    """Directory loaded detection dataset.

    Args:
        train_data_directory (str): path to training data directory.
        test_data_directory (str): path to test data directory.

    """

    image_feature_description: dict
    label_feature_description: dict

    def __init__(
            self,
            train_data_directory: str,
            test_data_directory: str,
            max_boxes: int = 5,
            **kwargs: Any) -> None:
        """Initilize params."""
        super(DirectoryObjectDitectionDataset, self).__init__(**kwargs)
        self.train_data_directory = pathlib.Path(train_data_directory)
        self.test_data_directory = pathlib.Path(test_data_directory)
        self.max_boxes = max_boxes
        self.x_train: List
        self.x_test: List

    def _resize_and_relative(
            self,
            image: np.array,
            box: np.array) -> Tuple[np.array, np.array]:
        """Resize and processing images."""
        def _process(image, box):
            _box = box.numpy()
            width = image.shape[1]
            height = image.shape[0]

            _box[:, 0] /= width
            _box[:, 2] /= width
            _box[:, 1] /= height
            _box[:, 3] /= height

            _box = _box[:self.max_boxes]
            if _box.shape[0] < self.max_boxes:
                zero_padding = np.zeros((self.max_boxes - _box.shape[0], 5), dtype=_box.dtype)
                _box = np.vstack((_box, zero_padding))
            return _box

        box = tf.py_function(_process, [image, box], [tf.float32])[0]
        image = tf.image.resize(image, self.input_shape[:2], tf.image.ResizeMethod.BILINEAR)
        image = image / 255.0
        return image, box

    def _decode_tfrecord(
            self,
            example_proto):
        image_feature, label_feature = tf.io.parse_single_sequence_example(
                example_proto,
                context_features=self.image_feature_description,
                sequence_features=self.label_feature_description)

        height = image_feature['height'][0]
        width = image_feature['width'][0]
        image = tf.io.decode_raw(image_feature['image'], tf.uint8)
        image = tf.reshape(image, tf.stack((height, width, 3)))
        label = label_feature['label']
        return image, label

    def _data_generator(
            self,
            data_path: pathlib.Path,
            repeat: bool = True) -> Generator:
        """Return training dataset.

        Args:
            data_path (pathlib.Path): data directory path.
            repeat (bool): whether or not to repeat data generate.

        Return:
            dataset (Generator): dataset generator

        """
        dataset = (
            tf.data.TFRecordDataset.list_files(str(data_path) + '/data*.tfrecord')
            .interleave(
                tf.data.TFRecordDataset,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .map(
                self._decode_tfrecord,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .map(
                self._resize_and_relative,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .cache()
            .shuffle(self.batch_size * 4)
            .batch(self.batch_size, drop_remainder=True)
        )

        if repeat:
            dataset = dataset.repeat()
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        while True:
            yield next(iter(dataset))

    def training_data_generator(self) -> Union[tf.keras.utils.Sequence, Generator]:
        """Return training dataset.

        Return:
            dataset (Union[tf.keras.utils.Sequence, Generator]): dataset generator

        """
        self.steps_per_epoch = len(self.x_train) // self.batch_size
        return self._data_generator(self.train_data_directory, repeat=True)

    def eval_data(self) -> Tuple[np.array, np.array]:
        """Return evaluation dataset.

        Return:
            dataset (Tuple[np.array, np.array]): evaluation dataset pair

        """
        images: List[np.array] = []
        boxes: List[np.array] = []
        generator = self._data_generator(self.test_data_directory, repeat=False)
        while True:
            try:
                _images, _boxes = next(generator)
                images.extend(_images)
                boxes.extend(_boxes)
            except StopIteration:
                break

        return np.array(images), np.array(boxes)

    def eval_data_generator(self) -> Union[tf.keras.utils.Sequence, Generator]:
        """Return test dataset.

        Return:
            dataset (Union[tf.keras.utils.Sequence, Generator]): dataset generator

        """
        self.eval_steps_per_epoch = len(self.x_test) // self.batch_size
        return self._data_generator(self.test_data_directory, repeat=True)


class ImageSegmentationDatasetBase(ImageClassifierDatasetBase):
    """Image segmentation dataset loader base class."""

    pass


class DirectoryImageSegmentationDataset(ImageSegmentationDatasetBase):
    """Directory loaded classifier dataset.

    Args:
        train_image_directory (str): path to training image directory.
        train_label_directory (str): path to training label directory.
        test_image_directory (str): path to test image directory.
        test_image_directory (str): path to test label directory.
        class_csv (str): path to class color definition csv. each rows has `name`, `r`, `g` and `b` columns.
        crop_width (int): crop width. if value is 0, we do not crop.
        crop_height (int): crop width. if value is 0, we do not crop.
        window_size (int): pre images and pot images scope size.
        sample_num_per_image (int): number of random sample per image.

    """

    def __init__(
            self,
            train_image_directory: str,
            train_label_directory: str,
            test_image_directory: str,
            test_label_directory: str,
            class_csv: str,
            crop_width: int = 0,
            crop_height: int = 0,
            window_size: int = 0,
            sample_num_per_image: int = 1,
            **kwargs: Any) -> None:
        """Initilize params."""
        super(DirectoryImageSegmentationDataset, self).__init__(**kwargs)
        self.train_image_directory = pathlib.Path(train_image_directory)
        self.train_label_directory = pathlib.Path(train_label_directory)
        self.test_image_directory = pathlib.Path(test_image_directory)
        self.test_label_directory = pathlib.Path(test_label_directory)
        self.class_dict: pd.DataFrame = pd.read_csv(class_csv)
        self.category_nums = len(self.class_dict)

        for path in self.train_image_directory.glob('**/*.*'):
            try:
                image = np.array(Image.open(path))
            except OSError:
                continue

            self.input_shape = image.shape
            break

        self.input_shape = (self.input_shape[0], self.input_shape[1], self.input_shape[2] * (2 * window_size + 1))
        if crop_height != 0 and crop_width != 0:
            self.input_shape = (crop_width, crop_height, self.input_shape[2])
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.window_size = window_size
        self.sample_num_per_image = sample_num_per_image

    def _random_crop(
            self,
            image: np.array,
            label: np.array) -> Tuple[np.array, np.array]:
        """Crop image and label to correspondence position.

        Args:
            image (np.array): target image.
            label (np.array): target label.

        Return:
            cropped_image (np.array): cropped image
            cropped_label (np.array): cropped label

        """
        if (image.shape[0] != label.shape[0]) or (image.shape[1] != label.shape[1]):
            raise ValueError('Image and label must have the same dimensions!')

        if self.crop_width == 0 or self.crop_height == 0:
            return image, label

        if (self.crop_width <= image.shape[0]) and (self.crop_height <= image.shape[1]):
            x = np.random.randint(0, image.shape[0]-self.crop_width)
            y = np.random.randint(0, image.shape[1]-self.crop_height)

            return image[x:x+self.crop_width, y:y+self.crop_height], label[x:x+self.crop_width, y:y+self.crop_height]
        else:
            raise Exception(
                'Crop shape (%d, %d) exceeds image dimensions (%d, %d)!'.format(
                    (self.crop_width, self.crop_height, image.shape[0], image.shape[1])))

    def _label_image_to_category(
            self,
            image: np.array) -> np.array:
        """Convert label image to category number list.

        Args:
            image (np.array): image array. shape is H x W x 3

        Return:
            label (np.array): label array. shape is H x W x category_nums

        """
        label = np.zeros((image.shape[0], image.shape[1], self.category_nums), dtype=np.bool)
        for i, r in self.class_dict.iterrows():
            equality = np.equal(image, [r['r'], r['g'], r['b']]).all(axis=-1)
            label[:, :, i][equality] = 1
        return label

    def _load_data(
            self,
            image_path: str,
            label_path: str) -> Tuple[np.array, np.array]:
        """Load one data.

        Args:
            image_path (str): path to image.
            label_path (str): path to correspondence label.

        Return:
            image (np.array): image object.
            label (mp.array): correspondence label object.
        """
        try:
            image = np.array(Image.open(image_path)) / 255.0
            label = self._label_image_to_category(np.array(Image.open(label_path).convert('RGB')))
            return image, label
        except OSError:
            return None, None

    def _data(
            self,
            image_path: pathlib.Path,
            label_path: pathlib.Path,
            max_length: Optional[int] = None) -> Tuple[np.array, np.array]:
        """Return training dataset.

        Args:
            image_path (pathlib.Path): image directory path
            label_path (pathlib.Path): label directory path

        Return:
            dataset (Tuple[np.array, np.array]): training dataset pair

        """
        image_paths = sorted(image_path.glob('**/*.*'))
        label_paths = sorted(label_path.glob('**/*.*'))
        if max_length:
            image_paths = random.choices(image_paths, k=max_length)
            label_paths = random.choices(label_paths, k=max_length)
        assert len(label_paths) == len(image_paths)

        pool = multiprocessing.pool.ThreadPool()
        results = []
        for image, label in zip(image_paths, label_paths):
            results.append(pool.apply_async(self._load_data, (image, label)))

        pool.close()
        pool.join()

        X: List = []
        y: List = []
        for res in results:
            image, label = res.get()
            if image is not None and label is not None:
                X.append(image)
                y.append(label)

        X_new = [np.concatenate(X[j-self.window_size:j+self.window_size+1], axis=2)
                 for j in range(self.window_size, len(y)-self.window_size)]
        y_new = [y[j] for j in range(self.window_size, len(y)-self.window_size)]
        X_new, y_new = zip(*[self._random_crop(image, label) for image, label in zip(X_new, y_new)])
        return np.array(X_new), np.array(y_new)

    def training_data(self) -> Tuple[np.array, np.array]:
        """Return training dataset.

        Return:
            dataset (Tuple[np.array, np.array]): training dataset pair

        """
        return self._data(self.train_image_directory, self.train_label_directory)

    def eval_data(self) -> Tuple[np.array, np.array]:
        """Return evaluation dataset.

        Return:
            dataset (Tuple[np.array, np.array]): evaluation dataset pair

        """
        return self._data(self.test_image_directory, self.test_label_directory, max_length=100)

    def _data_generator(
            self,
            image_path: pathlib.Path,
            label_path: pathlib.Path) -> Union[tf.keras.utils.Sequence, Generator]:
        """Return training dataset.

        Args:
            image_path (pathlib.Path): image directory path
            label_path (pathlib.Path): label directory path

        Return:
            dataset (Union[tf.keras.utils.Sequence, Generator]): dataset generator

        """
        image_directories = sorted([p for p in image_path.glob('*') if p.is_dir()])
        label_directories = sorted([p for p in label_path.glob('*') if p.is_dir()])

        while True:
            for im, la, in zip(image_directories, label_directories):
                image_paths = sorted(im.glob('*.*'))
                label_paths = sorted(la.glob('*.*'))
                if len(label_paths) != len(image_paths):
                    continue
                sample_num = len(image_paths) // self.sample_num_per_image
                indexes = np.random.choice(len(image_paths), sample_num)
                itr_num = int((sample_num - 2 * self.window_size) // (self.batch_size))

                for i in range(itr_num):
                    batch_ids = indexes[self.window_size + i * self.batch_size:2 * self.window_size + (i + 1) * self.batch_size]
                    X: List = []
                    y: List = []

                    for j in batch_ids:
                        image, label = self._load_data(image_paths[j], label_paths[j])
                        if image is not None and label is not None:
                            X.append(image)
                            y.append(label)

                    if self.window_size > 0:
                        X_new = [np.concatenate(X[j-self.window_size:j+self.window_size+1], axis=2)
                                 for j in range(self.window_size, len(y)-self.window_size)]
                        y_new = [y[j] for j in range(self.window_size, len(y)-self.window_size)]
                    else:
                        X_new, y_new = X, y
                    X_new, y_new = zip(*[self._random_crop(image, label) for image, label in zip(X_new, y_new)])
                    yield np.array(X_new), np.array(y_new)

    def training_data_generator(self) -> Union[tf.keras.utils.Sequence, Generator]:
        """Return training dataset.

        Return:
            dataset (Union[tf.keras.utils.Sequence, Generator]): dataset generator

        """
        self.steps_per_epoch = len(list(self.train_image_directory.glob('**/*.*'))) // self.batch_size // self.sample_num_per_image
        return self._data_generator(self.train_image_directory, self.train_label_directory)

    def eval_data_generator(self) -> Union[tf.keras.utils.Sequence, Generator]:
        """Return evaluation dataset.

        Return:
            dataset (Union[tf.keras.utils.Sequence, Generator]): dataset generator

        """
        self.eval_steps_per_epoch = len(list(self.test_image_directory.glob('*'))) // self.batch_size // self.sample_num_per_image
        return self._data_generator(self.test_image_directory, self.test_label_directory)


class TextDatasetBase(DatasetBase):
    """Image classification dataset loader base class.

    Args:
        batch_size (int): training batch size.

    """

    START_TOKEN = '<S>'
    END_TOKEN = '<E>'

    def __init__(
            self,
            batch_size: int = 32,
            seq_length: int = 128,
            vocab_size: int = 1000,
            **kwargs: Any) -> None:
        """Initialize data generator."""
        self.batch_size = batch_size
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, **kwargs)
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.steps_per_epoch: int
        self.eval_steps_per_epoch: int


class BinaryTextDataset(TextDatasetBase):
    """Memory loaded text dataset."""

    def __init__(self, **kwargs: Any) -> None:
        super(BinaryTextDataset, self).__init__(**kwargs)
        self.x_train: np.array
        self.x_test: np.array

    def training_data(self) -> Tuple[np.array, np.array]:
        """Return training dataset.

        Return:
            dataset (Tuple[np.array, np.array]): training dataset pair

        """
        self.tokenizer.fit_on_texts(self.x_train)
        sequences = self.tokenizer.texts_to_sequences(self.x_train)
        sequences = self.uniformed_sequences(sequences, length=self.seq_length+1)
        sequences = np.array(sequences)
        return sequences[:, :-1], sequences[:, 1:]

    def training_data_generator(self) -> Union[tf.keras.utils.Sequence, Generator]:
        """Return training dataset.

        Return:
            dataset (Union[tf.keras.utils.Sequence, Generator]): dataset generator

        """
        inputs, targets = self.training_data()
        self.steps_per_epoch = len(inputs) // self.batch_size
        dataset = tf.data.Dataset \
                    .from_tensor_slices((inputs, targets)) \
                    .shuffle(self.batch_size * 4, reshuffle_each_iteration=True) \
                    .repeat() \
                    .batch(self.batch_size, drop_remainder=True)
        return dataset

    def eval_data(self) -> Tuple[np.array, np.array]:
        """Return evaluation dataset.

        Return:
            dataset (Tuple[np.array, np.array]): evaluation dataset pair

        """
        sequences = self.tokenizer.texts_to_sequences(self.x_test)
        sequences = self.uniformed_sequences(sequences, length=self.seq_length+1)
        sequences = np.array(sequences)
        return sequences[:, :-1], sequences[:, 1:]

    def eval_data_generator(self) -> Union[tf.keras.utils.Sequence, Generator]:
        """Return evaluation dataset.

        Return:
            dataset (Union[tf.keras.utils.Sequence, Generator]): dataset generator

        """
        inputs, targets = self.eval_data()
        self.eval_steps_per_epoch = len(inputs) // self.batch_size
        dataset = tf.data.Dataset \
                    .from_tensor_slices((inputs, targets)) \
                    .batch(self.batch_size, drop_remainder=False)
        return dataset

    def uniformed_sequences(
            self,
            sequences: np.array,
            length: Optional[int] = None) -> np.array:
        """Return decode texts from word id sequences.

        Args:
            sequences (np.array): word id sequences.
            length (int): referenced length. default is self.seq_length.

        Return:
            padded_sequences (np.array]): word id sequences uniformed length.

        """
        if length is None:
            length = self.seq_length
        return tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post', truncating='post', maxlen=length)

    def decode(
            self,
            sequences: np.array) -> List[str]:
        """Return decode texts from word id sequences.

        Args:
            sequences (np.array): word id sequences.

        Return:
            texts (List[str]): decoded text.

        """
        return self.tokenizer.sequences_to_texts(sequences)

    def encode(
            self,
            texts: List[str]) -> np.array:
        """Return encode word id sequences from texts.

        Args:
            texts (List[str]): decoded text.

        Return:
            sequences (np.array): word id sequences.

        """
        return self.tokenizer.texts_to_sequences(texts)
