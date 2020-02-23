# Copyright 2019 Katsuya Shimabukuro. All rights reserved.
# Licensed under the MIT License.
import pathlib
import requests
import tqdm
from dataset.base import DirectoryObjectDitectionDataset

BASE_URL = 'http://images.cocodataset.org/zips/'
BASE_ANNOTATION_URL = 'http://images.cocodataset.org/annotations/'
TRAIN_FILE = 'train2014.zip'
EVAL_FILE = 'val2014.zip'
TEST_FILE = 'test2014.zip'
TRAIN_ANN_FILE = 'annotations_trainval2014.zip'
TEST_ANN_FILE = 'image_info_test2014.zip'


class MSCococDataset(DirectoryObjectDitectionDataset):
    """MSCOCO dataset loader."""
    pass


def download(
        artifact_directory: pathlib.Path,
        before_artifact_directory: pathlib.Path = None) -> None:
    """Download MSCOCO image and annotation data from cocodataset.org.

    Args:
        artifact_directory (Path): file save path.
        before_artifact_directory (Path): non use.

    """
    save_path = artifact_directory
    save_path.mkdir(parents=True, exist_ok=True)

    for f in [TRAIN_FILE, EVAL_FILE, TEST_FILE]:
        file_path = save_path.joinpath(f)
        res = requests.get(BASE_URL + f, stream=True)
        size = int(res.headers['Content-Length'])
        pbar = tqdm.tqdm(desc=str(f), total=size, unit='B', unit_scale=True)
        with file_path.open('wb') as w:
            for buf in res.iter_content(chunk_size=1024**2):
                w.write(buf)
                pbar.update(len(buf))
        res.close()

    for f in [TRAIN_ANN_FILE, TEST_ANN_FILE]:
        file_path = save_path.joinpath(f)
        res = requests.get(BASE_ANNOTATION_URL + f, stream=True)
        size = int(res.headers['Content-Length'])
        pbar = tqdm.tqdm(desc=str(f), total=size, unit='B', unit_scale=True)
        with file_path.open('wb') as w:
            for buf in res.iter_content(chunk_size=1024**2):
                w.write(buf)
                pbar.update(len(buf))
        res.close()
