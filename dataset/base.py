from typing import Tuple, List, Sequence, Any, Iterator
import tensorflow as tf


class DatasetBase(object):
    """Dataset loader base class."""

    def __init__(self) -> None:
        """Load data and setup preprocessing."""
        pass

    def training_data(self) -> Tuple[List, List]:
        """Return training dataset.

        Return:
            dataset (Tuple[List, List]): training dataset pair

        """
        pass

    def training_data_generator(self) -> Iterator:
        """Return training dataset iterator.

        Return:
            dataset (Iterator): dataset iterator

        """
        pass

    def eval_data(self) -> Tuple[List, List]:
        """Return evaluation dataset.

        Return:
            dataset (Tuple[List, List]): evaluation dataset pair

        """
        pass


class ImageClassifierDatasetBase(DatasetBase):
    """Image classification dataset loader base class.

    Args:
        batch_size (int): training batch size.

    """

    input_shape: Sequence[int] = (1, 1)
    category_nums: int = 1
    steps_per_epoch: int = 1

    def __init__(
            self,
            batch_size: int = 32,
            **kwargs: Any
            ) -> None:
        """Initialize data generator."""
        self.batch_size = batch_size
        self.train_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(**kwargs)
        self.eval_data_gen = tf.keras.preprocessing.image.ImageDataGenerator()


class BinaryImageClassifierDataset(ImageClassifierDatasetBase):
    """Memory loaded classifier dataset."""

    def __init__(self, **kwargs: Any) -> None:
        super(BinaryImageClassifierDataset, self).__init__(**kwargs)
        self.x_train: List = [None]
        self.x_test: List = [None]
        self.y_train: List = [None]
        self.y_test: List = [None]

    def training_data(self) -> Tuple[List, List]:
        """Return training dataset.

        Return:
            dataset (Tuple[List, List]): training dataset pair

        """
        return (self.train_data_gen.random_transform(self.x_train), self.y_train)

    def training_data_generator(self) -> Iterator:
        """Return training dataset.

        Return:
            dataset (Iterator): dataset generator

        """
        self.steps_per_epoch = len(self.x_train) // self.batch_size
        return self.train_data_gen.flow(self.x_train, y=self.y_train, batch_size=self.batch_size)

    def eval_data(self) -> Tuple[List, List]:
        """Return evaluation dataset.

        Return:
            dataset (Tuple[List, List]): evaluation dataset pair

        """
        return (self.x_test, self.y_test)
