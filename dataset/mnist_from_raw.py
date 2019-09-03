from typing import Any, List
import pathlib
import gzip
import io
import requests
import tqdm
import numpy as np
from dataset.base import BinaryImageClassifierDataset


mnist_files = [
    'train-images-idx3-ubyte',
    'train-labels-idx1-ubyte',
    't10k-images-idx3-ubyte',
    't10k-labels-idx1-ubyte']

MNIST_FILE_NAME = 'mnist.npz'


class MnistFromRawDataset(BinaryImageClassifierDataset):
    """Mnist dataset loader from  from original homepage.

    Args:
        data_path (pathlib.Path): path to data file directory.
        data_normalize_style (str): data noramlization style. Allowed valus are 0to1 ot -1to1

    """

    input_shape = (28, 28, 1)
    category_nums = 10

    def __init__(
            self,
            data_path: pathlib.Path,
            data_normalize_style: str = '0to1',
            **kwargs: Any) -> None:
        """Load data and setup preprocessing."""
        super(MnistFromRawDataset, self).__init__(**kwargs)
        with np.load(str(data_path.joinpath(MNIST_FILE_NAME))) as f:
            x_train = f[mnist_files[0]]
            y_train = f[mnist_files[1]]
            x_test = f[mnist_files[2]]
            y_test = f[mnist_files[3]]

        if data_normalize_style == '0to1':
            x_train, x_test = x_train / 255.0, x_test / 255.0
        elif data_normalize_style == '-1to1':
            x_train, x_test = (x_train - 127.5) / 127.5, (x_test - 127.5) / 127.5
        else:
            raise ValueError('Data normaliztion style: {} is not supported.'.format(data_normalize_style))

        x_train = np.expand_dims(x_train, axis=3)
        x_test = np.expand_dims(x_test, axis=3)

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test


def download_data(
        artifact_directory: pathlib.Path,
        before_artifact_directory: pathlib.Path) -> None:
    """Download mnist data from original homepage.

    Args:
        artifact_directory (pathlib.Path): path to save directory.

    """
    for path in mnist_files:
        path += '.gz'
        file_url = 'http://yann.lecun.com/exdb/mnist/' + path
        file_size = int(requests.head(file_url).headers["content-length"])
        res = requests.get(file_url, stream=True)
        pbar = tqdm.tqdm(total=file_size, unit="B", unit_scale=True, desc=path)
        with artifact_directory.joinpath(path).open('wb') as f:
            for chunk in res.iter_content(chunk_size=1024*100):
                f.write(chunk)
                pbar.update(len(chunk))
        res.close()


def decompose_data(
        artifact_directory: pathlib.Path,
        before_artifact_directory: pathlib.Path) -> None:
    """Download mnist data from original homepage.

    Args:
        artifact_directory (pathlib.Path): path to save directory.
        before_artifact_directory (pathlib.Path): path to before save directory.

    """
    output = {}
    for path in mnist_files:
        with gzip.open(str(before_artifact_directory.joinpath(path+'.gz')), 'rb') as f:
            binary = io.BytesIO(f.read())
        magic = int.from_bytes(binary.read(4), byteorder='big')
        size = int.from_bytes(binary.read(4), byteorder='big')
        objs: List[np.array] = []
        if magic == 2051:
            # case image
            H = int.from_bytes(binary.read(4), byteorder='big')
            W = int.from_bytes(binary.read(4), byteorder='big')
            for _ in range(size):
                objs.append(np.frombuffer(binary.read(H*W), dtype=np.uint8).reshape((H, W)))
        elif magic == 2049:
            # case label
            for _ in range(size):
                objs.append(np.frombuffer(binary.read(1), dtype=np.uint8))
        else:
            binary.close()
            raise ValueError('Unsupported magic number {}.'.format(magic))
        binary.close()
        output[path] = np.array(objs)
    np.savez(str(artifact_directory.joinpath(MNIST_FILE_NAME)), **output)
