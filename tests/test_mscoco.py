# Copyright 2020 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
import shutil
import pathlib
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
