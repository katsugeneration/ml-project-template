# Copyright 2020 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
import subprocess
import shutil
import pathlib
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
from nose.tools import ok_, eq_, raises
from dataset.mscoco import MSCococDatectionDataset, convert_tfrecord, image_feature_description, label_feature_description


class TestMSCococDatectionDataset(object):
    def test_init(self):
        dataset = MSCococDatectionDataset(
                train_image_directory=pathlib.Path(__file__).parent.joinpath('data/mscoco/image'),
                train_label_path=pathlib.Path(__file__).parent.joinpath('data/mscoco/annotations.json'),
                test_image_directory=pathlib.Path(__file__).parent.joinpath('data/mscoco/image'),
                test_label_path=pathlib.Path(__file__).parent.joinpath('data/mscoco/annotations.json'))
        ok_(len(dataset.x_train) != 0)
        ok_(isinstance(dataset.x_train[0], pathlib.Path))
        ok_(len(dataset.y_train) != 0)
        ok_(isinstance(dataset.y_train[0], list))
        ok_(len(dataset.y_train[0]) != 0)

    def test_convert_tfrecord(self):
        path = pathlib.Path(__file__).parent.joinpath('after_path')
        if not path.exists():
            path.mkdir(parents=True)
        convert_tfrecord(path, pathlib.Path(__file__).parent.joinpath('data/mscoco/'))

        ok_(path.joinpath('train.tfrecord').exists())
        ok_(path.joinpath('test.tfrecord').exists())

        dataset = tf.data.TFRecordDataset([str(path.joinpath('train.tfrecord'))])
        def _parse_image_function(example_proto):
            return tf.io.parse_single_sequence_example(
                example_proto,
                context_features=image_feature_description,
                sequence_features=label_feature_description)
        for raw_record in dataset.map(_parse_image_function).take(1):
            image_feature, label_feature = raw_record
        image = Image.frombytes('RGB', (image_feature['width'].numpy()[0], image_feature['height'].numpy()[0]), image_feature['image'].numpy())
        image.save('test.png', "PNG")
        subprocess.run(('open test.png'), shell=True)
        label = label_feature['label'].numpy()
        ok_((np.abs(label - [[493.015, 208.61, 214.97, 297.16, 25], [119.025, 384.085, 132.03, 55.19, 25]]) <= 0.0001).all())
        shutil.rmtree(path)
        pathlib.Path('test.png').unlink()

    def test_train_generator(self):
        dataset = MSCococDatectionDataset(
                train_image_directory=pathlib.Path(__file__).parent.joinpath('data/mscoco/image'),
                train_label_path=pathlib.Path(__file__).parent.joinpath('data/mscoco/annotations.json'),
                test_image_directory=pathlib.Path(__file__).parent.joinpath('data/mscoco/image'),
                test_label_path=pathlib.Path(__file__).parent.joinpath('data/mscoco/annotations.json'),
                batch_size=2)
        generator = dataset.training_data_generator()
        image, box = next(generator)
        eq_(len(image), 2)
        eq_(len(box[0]), dataset.max_boxes)
        image, box = next(generator)
        eq_(len(image), 2)

    def test_eval_generator(self):
        dataset = MSCococDatectionDataset(
                train_image_directory=pathlib.Path(__file__).parent.joinpath('data/mscoco/image'),
                train_label_path=pathlib.Path(__file__).parent.joinpath('data/mscoco/annotations.json'),
                test_image_directory=pathlib.Path(__file__).parent.joinpath('data/mscoco/image'),
                test_label_path=pathlib.Path(__file__).parent.joinpath('data/mscoco/annotations.json'),
                batch_size=2)
        generator = dataset.eval_data_generator()
        image, box = next(generator)
        eq_(len(image), 2)
        eq_(len(box[0]), dataset.max_boxes)
        image, box = next(generator)
        eq_(len(image), 2)

    def test_draw_bbox(self):
        dataset = MSCococDatectionDataset(
                train_image_directory=pathlib.Path(__file__).parent.joinpath('data/mscoco/image'),
                train_label_path=pathlib.Path(__file__).parent.joinpath('data/mscoco/annotations.json'),
                test_image_directory=pathlib.Path(__file__).parent.joinpath('data/mscoco/image'),
                test_label_path=pathlib.Path(__file__).parent.joinpath('data/mscoco/annotations.json'))

        image = Image.open(dataset.x_train[0])
        draw = ImageDraw.Draw(image)
        for box in dataset.y_train[0]:
            x, y, w, h, category = box
            draw.rectangle(((x - w / 2, y - h / 2), (x + w / 2, y + h / 2)), outline='red', width=5)
        image.save('test.png', "PNG")
        subprocess.run(('open test.png'), shell=True)
        pathlib.Path('test.png').unlink()

    def test_draw_resized_bbox(self):
        dataset = MSCococDatectionDataset(
                train_image_directory=pathlib.Path(__file__).parent.joinpath('data/mscoco/image'),
                train_label_path=pathlib.Path(__file__).parent.joinpath('data/mscoco/annotations.json'),
                test_image_directory=pathlib.Path(__file__).parent.joinpath('data/mscoco/image'),
                test_label_path=pathlib.Path(__file__).parent.joinpath('data/mscoco/annotations.json'),
                batch_size=2)

        image, boxes = next(dataset.training_data_generator())
        image = Image.fromarray((image[0] * 255.0).astype(np.uint8))
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
