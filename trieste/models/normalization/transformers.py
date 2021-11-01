from abc import ABC, abstractmethod
import tensorflow as tf
from trieste.types import TensorType

class DataTransformer(ABC):
    """A data transformer."""

    def __init__(self, data: TensorType) -> None:
        """Initialize object and set parameters of the transform."""
        super().__init__()
        self.set_parameters(data)

    @abstractmethod
    def set_parameters(self, data: TensorType) -> None:
        """Set parameters of the transformation based on the unnormalized dataset.

        :param data: The unnomralized data.
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, data: TensorType) -> TensorType:
        """Forward transformation of data (from unnormalized to normalized).

        :param data: The unnormalized data.
        :return: The normalized data. Should be the same shape as the input data.
        """
        raise NotImplementedError

    @abstractmethod
    def inverse_transform(self, data: TensorType) -> TensorType:
        """Inverse transformation of data (from normalized to unnormalized).

        :param data: The normalized data.
        :return: The unnormalized data. Should be the same shape as the input data.
        """
        raise NotImplementedError

    def transform_variance(self, variance: TensorType) -> TensorType:
        """[Optionally] transform the variance over the unnormalised data to the variance over the
        normalized data."""
        return variance

    def inverse_transform_variance(self, variance: TensorType) -> TensorType:
        """[Optionally] transform the variance over the normalised data to the variance over the
        unnormalized data."""
        return variance


class IdentityTransformer(DataTransformer):
    """Passes data through, unchanged."""

    def set_parameters(self, data: TensorType) -> None:
        pass

    def transform(self, data: TensorType) -> TensorType:
        return data

    def inverse_transform(self, data: TensorType) -> TensorType:
        return data


class StandardTransformer(DataTransformer):
    """Transforms data to zero mean and unit variance."""

    def __init__(self, data: TensorType) -> None:
        self._mean: TensorType = None
        self._std: TensorType = None
        super().__init__(data)

    def set_parameters(self, data: TensorType) -> None:
        self._mean = tf.squeeze(tf.math.reduce_mean(data, axis=0))
        self._std = tf.squeeze(tf.math.reduce_std(data, axis=0))

    def transform(self, data: TensorType) -> TensorType:
        return (data - self._mean) / self._std

    def inverse_transform(self, data: TensorType) -> TensorType:
        return data * self._std + self._mean

    def transform_variance(self, variance: TensorType) -> TensorType:
        return variance / self._std ** 2

    def inverse_transform_variance(self, variance: TensorType) -> TensorType:
        return variance * self._std ** 2


class MinMaxTransformer(DataTransformer):
    """Transforms data to the unit cube, i.e. to the interval [0, 1] for all dimensions."""

    def __init__(self, data: TensorType) -> None:
        self._min: TensorType = None
        self._delta: TensorType = None
        super().__init__(data)

    def set_parameters(self, data: TensorType) -> None:
        self._min = tf.squeeze(tf.math.reduce_min(data, axis=0))
        max_data = tf.squeeze(tf.math.reduce_max(data, axis=0))
        self._delta = max_data - self._min

    def transform(self, data: TensorType) -> TensorType:
        return (data - self._min) / self._delta

    def inverse_transform(self, data: TensorType) -> TensorType:
        return data * self._delta + self._min

    def transform_variance(self, variance: TensorType) -> TensorType:
        return variance / self._delta ** 2

    def inverse_transform_variance(self, variance: TensorType) -> TensorType:
        return variance * self._delta ** 2
