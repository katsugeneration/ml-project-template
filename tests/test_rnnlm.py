# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from nose.tools import ok_, eq_
import numpy as np
from model.rnnlm import RNNLM
from tests.utils.dummy_dataset import TextDummyDataset


class TestRNNLM(object):
    def test_init(self):
        rnnlm = RNNLM(
                    dataset=TextDummyDataset(
                        np.array([]),
                        vocab_size=100),
                    embedding_dim=16)
        eq_(rnnlm._embdeeing.embeddings.shape[0], 100)
        eq_(rnnlm._embdeeing.embeddings.shape[1], 16)

    def test_train(self):
        rnnlm = RNNLM(
                    dataset=TextDummyDataset(
                        np.array([' '.join(['sss'] * 128)] * 32),
                        vocab_size=100),
                    seq_length=128,
                    batch_size=32,
                    embedding_dim=16)
        history = rnnlm.train()

        ok_('loss' in history)

    def test_train_with_short_sequences(self):
        rnnlm = RNNLM(
                    dataset=TextDummyDataset(
                        np.array([' '.join(['sss'] * (128 - 5)), ' '.join(['aaa'] * (128))] * 32),
                        vocab_size=100),
                    seq_length=128,
                    batch_size=32,
                    embedding_dim=16)
        history = rnnlm.train()

        ok_('loss' in history)

    def test_mask(self):
        rnnlm = RNNLM(
                    dataset=TextDummyDataset(
                        np.array([' '.join(['sss'] * 128)] * 32),
                        vocab_size=100),
                    seq_length=128,
                    batch_size=32,
                    embedding_dim=16)
        outputs = rnnlm.model(np.array([[1] * 30 + [0] * 98], dtype=np.float32))

        ok_(np.any(outputs[:, 0] != outputs[:, 1]))
        ok_(np.all(outputs[:, 30] == outputs[:, 31]))
