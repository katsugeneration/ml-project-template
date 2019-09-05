# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from nose.tools import ok_, eq_
from dataset.base import DirectoryImageSegmentationDataset


class TestDirectoryImageSegmentationDataset(object):
    def test_init(self):
        DirectoryImageSegmentationDataset(
                train_image_directory='tests/CamVid/train',
                train_label_directory='tests/CamVid/train_labels',
                test_image_directory='tests/CamVid/val',
                test_label_directory='tests/CamVid/val_labels',
                class_csv='tests/CamVid/class_dict.csv')

    def test_get_train_data(self):
        dataset = DirectoryImageSegmentationDataset(
                batch_size=8,
                train_image_directory='tests/CamVid/train',
                train_label_directory='tests/CamVid/train_labels',
                test_image_directory='tests/CamVid/val',
                test_label_directory='tests/CamVid/val_labels',
                class_csv='tests/CamVid/class_dict.csv')
        x, y = dataset.training_data()
        eq_(x.shape[1:3], (720, 960))
        eq_(x[0].shape[:2], y[0].shape[:2])
        eq_(len(y[0].shape), 3)

    def test_get_train_data_generator(self):
        dataset = DirectoryImageSegmentationDataset(
                batch_size=8,
                train_image_directory='tests/CamVid/train',
                train_label_directory='tests/CamVid/train_labels',
                test_image_directory='tests/CamVid/val',
                test_label_directory='tests/CamVid/val_labels',
                class_csv='tests/CamVid/class_dict.csv')
        generator = dataset.training_data_generator()
        x, y = next(generator)
        eq_(x.shape[0], 8)
        eq_(x.shape[1:3], (720, 960))
        eq_(x[0].shape[:2], y[0].shape[:2])
        eq_(len(y[0].shape), 3)
        eq_(y[0].shape[2], dataset.category_nums)

    def test_get_train_data_generator_with_crop(self):
        dataset = DirectoryImageSegmentationDataset(
                batch_size=8,
                train_image_directory='tests/CamVid/train',
                train_label_directory='tests/CamVid/train_labels',
                test_image_directory='tests/CamVid/val',
                test_label_directory='tests/CamVid/val_labels',
                class_csv='tests/CamVid/class_dict.csv',
                crop_height=480,
                crop_width=320)
        generator = dataset.training_data_generator()
        x, y = next(generator)
        eq_(x.shape[0], 8)
        eq_(x.shape[1:3], (320, 480))
        eq_(x[0].shape[:2], y[0].shape[:2])
        eq_(len(y[0].shape), 3)
        eq_(y[0].shape[2], dataset.category_nums)

    def test_get_eval_data_generator(self):
        dataset = DirectoryImageSegmentationDataset(
                batch_size=4,
                train_image_directory='tests/CamVid/train',
                train_label_directory='tests/CamVid/train_labels',
                test_image_directory='tests/CamVid/val',
                test_label_directory='tests/CamVid/val_labels',
                class_csv='tests/CamVid/class_dict.csv')
        generator = dataset.eval_data_generator()
        x, y = next(generator)
        eq_(x.shape[0], 4)
        eq_(x.shape[1:3], (720, 960))
        eq_(x[0].shape[:2], y[0].shape[:2])
        eq_(len(y[0].shape), 3)
        eq_(y[0].shape[2], dataset.category_nums)
