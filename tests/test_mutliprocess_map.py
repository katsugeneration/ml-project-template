# Copyright 2020 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from nose.tools import ok_, eq_
import time
import numpy as np
from dataset.utils.multiprocess import Map


class TestMultiprocessMap(object):
    def _process(self, a, b):
        return a + b

    def test_init(self):
        map = Map(self._process, 3)

    def test_process(self):
        map = Map(self._process, 3)
        a = [np.ones((1, 2)) * i for i in range(10)]
        b = [np.ones((1, 2)) * i for i in range(10)]
        map.put(zip(a, b))
        time.sleep(1)
        ret = map.get(3)
        eq_(len(ret), 3)
        map.close()

    def test_get_before(self):
        map = Map(self._process, 3)
        ret = map.get(3)
        eq_(len(ret), 0)
        map.close()

    def test_get_lower(self):
        map = Map(self._process, 3)
        a = [np.ones((1, 2)) * i for i in range(2)]
        b = [np.ones((1, 2)) * i for i in range(2)]
        map.put(zip(a, b))
        time.sleep(1)
        ret = map.get(3)
        eq_(len(ret), 2)
        map.close()
