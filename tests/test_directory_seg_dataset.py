# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from nose.tools import ok_, eq_
from dataset.base import DirectoryImageSegmentationDataset


class TestDirectoryImageSegmentationDataset(object):
    def test_init(self):
        dataset = DirectoryImageSegmentationDataset(
                directory='tests/CamVid',
                class_csv='tests/CamVid/class_dict.csv')

    def test_get_train_data(self):
        dataset = DirectoryImageSegmentationDataset(
                directory='tests/CamVid',
                class_csv='tests/CamVid/class_dict.csv')
        x, y = dataset.training_data()
        eq_(x[0].shape[:2], y[0].shape[:2])
        eq_(len(y[0].shape), 3)
        ok_(y[0].shape[2], dataset.category_nums)

    def test_get_train_data_generator(self):
        dataset = DirectoryImageSegmentationDataset(
                batch_size=8,
                directory='tests/CamVid',
                class_csv='tests/CamVid/class_dict.csv')
        generator = dataset.training_data_generator()
        x, y = next(generator)
        eq_(x.shape[0], 8)
        eq_(x[0].shape[:2], y[0].shape[:2])
        eq_(len(y[0].shape), 3)
        eq_(y[0].shape[2], dataset.category_nums)