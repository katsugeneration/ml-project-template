# Copyright 2020 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
import subprocess
import shutil
import pathlib
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
from nose.tools import ok_, eq_, raises
from dataset.open_images import OpenImagesClassificationDataset


class TestOpenImagesClassificationDataset(object):
    def setup(self):
        self.path = pathlib.Path(__file__).parent.joinpath('after_path')
        self.path.mkdir(parents=True, exist_ok=True)

    def teardown(self):
        shutil.rmtree(self.path)

    def test_init(self):
        OpenImagesClassificationDataset.convert_tfrecord(self.path, pathlib.Path(__file__).parent.joinpath('data/openimages/'))
        dataset = OpenImagesClassificationDataset(
                train_image_directory=pathlib.Path(__file__).parent.joinpath('data/openimages/train'),
                test_image_directory=pathlib.Path(__file__).parent.joinpath('data/openimages/validation'),
                train_data_directory=self.path.joinpath('train'),
                test_data_directory=self.path.joinpath('test'),
                batch_size=2)
        ok_(len(dataset.x_train) != 0)
        ok_(isinstance(dataset.x_train[0], pathlib.Path))

    def test_convert_tfrecord(self):
        OpenImagesClassificationDataset.convert_tfrecord(self.path, pathlib.Path(__file__).parent.joinpath('data/openimages/'), split_num=1)

        ok_(self.path.joinpath('train/data0.tfrecord').exists())
        ok_(self.path.joinpath('test/data0.tfrecord').exists())

        dataset = tf.data.TFRecordDataset([str(self.path.joinpath('train/data0.tfrecord'))])
        def _parse_image_function(example_proto):
            return tf.io.parse_single_sequence_example(
                example_proto,
                context_features=OpenImagesClassificationDataset.image_feature_description,
                sequence_features=OpenImagesClassificationDataset.label_feature_description)
        for raw_record in dataset.map(_parse_image_function).take(1):
            image_feature, label_feature = raw_record
        image = Image.frombytes('RGB', (image_feature['width'].numpy()[0], image_feature['height'].numpy()[0]), image_feature['image'].numpy())
        image.save('test.png', "PNG")
        subprocess.run(('open test.png'), shell=True)
        label = label_feature['label'].numpy().flatten()
        eq_(len(label.shape), 1)
        pathlib.Path('test.png').unlink()

    def test_train_generator(self):
        OpenImagesClassificationDataset.convert_tfrecord(self.path, pathlib.Path(__file__).parent.joinpath('data/openimages/'), split_num=2)
        dataset = OpenImagesClassificationDataset(
                train_image_directory=pathlib.Path(__file__).parent.joinpath('data/openimages/train'),
                test_image_directory=pathlib.Path(__file__).parent.joinpath('data/openimages/validation'),
                train_data_directory=self.path.joinpath('train'),
                test_data_directory=self.path.joinpath('test'),
                batch_size=2)

        generator = dataset.training_data_generator()
        image, label = next(generator)
        eq_(len(image), 2)
        eq_(image.shape[1:], dataset.input_shape)
        eq_(len(label[0]), dataset.category_nums)
        image, box = next(generator)
        eq_(len(image), 2)
