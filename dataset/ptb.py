# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
from typing import Any
import pathlib
import requests
from dataset.base import BinaryTextDataset

BASE_URL = 'https://raw.githubusercontent.com/tmatha/lstm/master/'
TRAIN_FILE = 'ptb.train.txt'
EVAL_FILE = 'ptb.valid.txt'
TEST_FILE = 'ptb.test.txt'


class PtbDataset(BinaryTextDataset):
    """Penn Tree Banl dataset loader."""

    def __init__(
            self,
            path: pathlib.Path,
            **kwargs: Any) -> None:
        """Load data and setup preprocessing.

        Args:
            path (Path): file save path.

        """
        super(PtbDataset, self).__init__(**kwargs)

        with open(path.joinpath(TRAIN_FILE), 'r', encoding='utf-8') as f:
            self.x_train = [line.strip() for line in f]
        with open(path.joinpath(EVAL_FILE), 'r', encoding='utf-8') as f:
            self.x_test = [line.strip() for line in f]


def download(
        artifact_directory: pathlib.Path,
        before_artifact_directory: pathlib.Path = None) -> None:
    """Download pptb text data from github.

    Args:
        artifact_directory (Path): file save path.
        before_artifact_directory (Path): non use.

    """
    save_path = artifact_directory
    save_path.mkdir(parents=True, exist_ok=True)

    for f in [TRAIN_FILE, EVAL_FILE, TEST_FILE]:
        file_path = save_path.joinpath(f)
        res = requests.get(BASE_URL + f, stream=True)
        with file_path.open('wb') as w:
            for buf in res.iter_content(chunk_size=1024**2):
                w.write(buf)
        res.close()
