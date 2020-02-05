# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
import shutil
import pathlib
from nose.tools import ok_, eq_
from dataset.ptb import PtbDataset, download, TEST_FILE


class TestPtbDataset(object):
    def test_download(self):
        path = pathlib.Path('tmp')
        download(path)
        ok_(path.exists())
        ok_(path.joinpath(TEST_FILE).exists())
        shutil.rmtree(path)

    def test_init(self):
        path = pathlib.Path('tmp')
        download(path)
        dataset = PtbDataset(path)
        ok_(len(dataset.x_train) != 0)
        shutil.rmtree(path)
