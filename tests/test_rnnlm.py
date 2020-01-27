# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from nose.tools import ok_, eq_
from model.rnnlm import RNNLM
from tests.utils.dummy_dataset import TextDummyDataset


class TestFCNN(object):
    def test_init(self):
        rnnlm = RNNLM(
                    dataset=TextDummyDataset(vocab_size=100),
                    embedding_dim=16)
        eq_(rnnlm._embdeeing.embeddings.shape[0], 100)
        eq_(rnnlm._embdeeing.embeddings.shape[1], 16)

    def test_train(self):
        rnnlm = RNNLM(
                    dataset=TextDummyDataset(vocab_size=100),
                    embedding_dim=16)
        history = rnnlm.train()

        ok_('accuracy' in history)
        ok_('loss' in history)
