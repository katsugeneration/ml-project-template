# Copyright 2020 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
import subprocess
import shutil
import pathlib
from PIL import Image, ImageDraw
import numpy as np
from nose.tools import ok_, eq_
from dataset.mscoco import MSCococDatectionDataset


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

    def test_train_generator(self):
        dataset = MSCococDatectionDataset(
                train_image_directory=pathlib.Path(__file__).parent.joinpath('data/mscoco/image'),
                train_label_path=pathlib.Path(__file__).parent.joinpath('data/mscoco/annotations.json'),
                test_image_directory=pathlib.Path(__file__).parent.joinpath('data/mscoco/image'),
                test_label_path=pathlib.Path(__file__).parent.joinpath('data/mscoco/annotations.json'))
        image, box = next(dataset.training_data_generator())

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
                test_label_path=pathlib.Path(__file__).parent.joinpath('data/mscoco/annotations.json'))

        image, boxes = next(dataset.training_data_generator())
        image = Image.fromarray((image * 255.0).astype(np.uint8))
        draw = ImageDraw.Draw(image)
        for box in boxes:
            x, y, w, h, category = box
            x *= image.width
            w *= image.width
            y *= image.height
            h *= image.height
            draw.rectangle(((x - w / 2, y - h / 2), (x + w / 2, y + h / 2)), outline='red', width=5)
        image.save('test.png', "PNG")
        subprocess.run(('open test.png'), shell=True)
        pathlib.Path('test.png').unlink()
