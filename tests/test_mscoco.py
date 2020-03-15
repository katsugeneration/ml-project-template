# Copyright 2020 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
import subprocess
import shutil
import pathlib
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
from nose.tools import ok_, eq_, raises
from dataset.mscoco import MSCococDatectionDataset, convert_tfrecord


class TestMSCococDatectionDataset(object):
    def setup(self):
        self.path = pathlib.Path(__file__).parent.joinpath('after_path')
        self.path.mkdir(parents=True, exist_ok=True)

    def teardown(self):
        shutil.rmtree(self.path)

    def test_init(self):
        convert_tfrecord(self.path, pathlib.Path(__file__).parent.joinpath('data/mscoco/'))
        dataset = MSCococDatectionDataset(
                train_image_directory=pathlib.Path(__file__).parent.joinpath('data/mscoco/train2014'),
                test_image_directory=pathlib.Path(__file__).parent.joinpath('data/mscoco/eval2014'),
                train_data_directory=self.path.joinpath('train'),
                test_data_directory=self.path.joinpath('test'),
                batch_size=2)
        ok_(len(dataset.x_train) != 0)
        ok_(isinstance(dataset.x_train[0], pathlib.Path))

    def test_convert_tfrecord(self):
        convert_tfrecord(self.path, pathlib.Path(__file__).parent.joinpath('data/mscoco/'))

        ok_(self.path.joinpath('train/data.tfrecord').exists())
        ok_(self.path.joinpath('test/data.tfrecord').exists())

        dataset = tf.data.TFRecordDataset([str(self.path.joinpath('train/data.tfrecord'))])
        def _parse_image_function(example_proto):
            return tf.io.parse_single_sequence_example(
                example_proto,
                context_features=MSCococDatectionDataset.image_feature_description,
                sequence_features=MSCococDatectionDataset.label_feature_description)
        for raw_record in dataset.map(_parse_image_function).take(1):
            image_feature, label_feature = raw_record
        image = Image.frombytes('RGB', (image_feature['width'].numpy()[0], image_feature['height'].numpy()[0]), image_feature['image'].numpy())
        image.save('test.png', "PNG")
        subprocess.run(('open test.png'), shell=True)
        label = label_feature['label'].numpy()
        ok_((np.abs(label - [[493.015, 208.61, 214.97, 297.16, 25], [119.025, 384.085, 132.03, 55.19, 25]]) <= 0.0001).all())
        pathlib.Path('test.png').unlink()

    def test_train_generator(self):
        convert_tfrecord(self.path, pathlib.Path(__file__).parent.joinpath('data/mscoco/'))
        dataset = MSCococDatectionDataset(
                train_image_directory=pathlib.Path(__file__).parent.joinpath('data/mscoco/train2014'),
                test_image_directory=pathlib.Path(__file__).parent.joinpath('data/mscoco/eval2014'),
                train_data_directory=self.path.joinpath('train'),
                test_data_directory=self.path.joinpath('test'),
                batch_size=2)

        generator = dataset.training_data_generator()
        image, box = next(generator)
        eq_(len(image), 2)
        eq_(len(box[0]), dataset.max_boxes)
        image, box = next(generator)
        eq_(len(image), 2)

    def test_eval_generator(self):
        convert_tfrecord(self.path, pathlib.Path(__file__).parent.joinpath('data/mscoco/'))
        dataset = MSCococDatectionDataset(
                train_image_directory=pathlib.Path(__file__).parent.joinpath('data/mscoco/train2014'),
                test_image_directory=pathlib.Path(__file__).parent.joinpath('data/mscoco/eval2014'),
                train_data_directory=self.path.joinpath('train'),
                test_data_directory=self.path.joinpath('test'),
                batch_size=2)

        generator = dataset.eval_data_generator()
        image, box = next(generator)
        eq_(len(image), 2)
        eq_(len(box[0]), dataset.max_boxes)
        image, box = next(generator)
        eq_(len(image), 2)

    def test_draw_resized_bbox(self):
        convert_tfrecord(self.path, pathlib.Path(__file__).parent.joinpath('data/mscoco/'))
        dataset = MSCococDatectionDataset(
                train_image_directory=pathlib.Path(__file__).parent.joinpath('data/mscoco/train2014'),
                test_image_directory=pathlib.Path(__file__).parent.joinpath('data/mscoco/eval2014'),
                train_data_directory=self.path.joinpath('train'),
                test_data_directory=self.path.joinpath('test'),
                batch_size=2)

        image, boxes = next(dataset.training_data_generator())
        image = Image.fromarray((image.numpy()[0] * 255.0).astype(np.uint8))
        draw = ImageDraw.Draw(image)
        for box in boxes[0]:
            x, y, w, h, category = box
            x *= image.width
            w *= image.width
            y *= image.height
            h *= image.height
            draw.rectangle(((x - w / 2, y - h / 2), (x + w / 2, y + h / 2)), outline='red', width=5)
        image.save('test.png', "PNG")
        subprocess.run(('open test.png'), shell=True)
        pathlib.Path('test.png').unlink()
