# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
import shutil
from nose.tools import ok_, eq_
from dataset.ptb import PtbDataset, download, SAVE_PATH, TEST_FILE


class TestPtbDataset(object):
    def test_download(self):
        download()
        ok_(SAVE_PATH.exists())
        ok_(SAVE_PATH.joinpath(TEST_FILE).exists())
        shutil.rmtree(SAVE_PATH)

    def test_init(self):
        download()
        dataset = PtbDataset()
        ok_(len(dataset.x_train) != 0)
        shutil.rmtree(SAVE_PATH)
