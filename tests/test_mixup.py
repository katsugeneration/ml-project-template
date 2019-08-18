# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from nose.tools import ok_, eq_
import numpy as np
from dataset.mixup import MixupGenerator


class TestMixup(object):
    def test_init(self):
        gen = MixupGenerator(
                    np.zeros((32, 32, 32, 3)),
                    np.zeros((32, )),
                    batch_size=16)
        eq_(len(gen), 1)

    def test_generate(self):
        gen = MixupGenerator(
                    np.zeros((32, 32, 32, 3)),
                    np.zeros((32, )),
                    batch_size=16)

        
