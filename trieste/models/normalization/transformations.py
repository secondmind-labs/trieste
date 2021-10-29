from abc import ABC, abstractmethod
import tensorflow as tf
from trieste.types import TensorType

class DataTransformer(ABC):
    """A data transformer."""

    def __init__(self: data: TensorType) -> None:
        """Initialize object and set parameters of the transform."""
        super().__init__()
        self._set_parameters(data)

    @abstractmethod
    def _set_parameters(data: TensorType) -> None:
        """Set parameters of the transformation based on the unnormalized dataset.
        
        :param data: The unnomralized data.
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, data: TensorType) -> TensorType:
        """Forward transformation of data (from unnormalized to normalized).

        :param data: The unnormalized data.
        :param transformation_parameters: parameters of the transformation.
        :return: The normalized data. Should be the same shape as the input data.
        """
        raise NotImplementedError

    @abstractmethod
    def inverse_transform(self, data: TensorType) -> TensorType:
        """Inverse transformation of data (from normalized to unnormalized).

        :param data: The normalized data.
        :param transformation_parameters: parameters of the transformation.
        :return: The unnormalized data. Should be the same shape as the input data.
        """
        raise NotImplementedError

    def transform_covariance(self, covariance: TensorType) -> TensorType:
        """[Optionally] transform the covariance matrix over the unnormalised data to the covariance over the normalized data."""
        return covariance

    def inverse_transform_covariance(self, covariance: TensorType) -> TensorType:
        """[Optionally] transform the covariance matrix over the normalised data to the covariance over the unnormalized data."""
        return covariance


class IdentityTransformer(DataTransformer):
    """Passes data through, unchanged."""

    def transform(self, data: TensorType, **transformation_parameters) -> TensorType:
        return data

    def inverse_transform(self, data: TensorType, **transformation_parameters) -> TensorType:
        return data


class StandardTransformer(DataTransformer):
    """Transforms data to zero mean and unit variance."""

    def __init__(self, data: TensorType) -> None:
        super().__init__()
        self._mean: TensorType = None
        self._std: TensorType = None
        self._set_parameters(data)

    def _set_parameters(self, data: TensorType) -> TensorType:
        self._mean = tf.math.reduce_mean(data, axis=0)
        self._std = tf.math.reduce_std(data, axis=0)

    def transform(self, data: TensorType) -> TensorType:
        return (data - self._mean) / self._std

    def inverse_transform(self, data: TensorType) -> TensorType:
        return data * self._std + self._mean

    def transform_covariance(self, covariance: TensorType) -> TensorType:
        return covariance / self._std ** 2

    def inverse_transform_covariance(self, covariance: TensorType) -> TensorType:
        return covariance * self._std ** 2


class MinMaxTransformer(DataTransformer):
    """Transforms data to the unit cube, i.e. to the interval [0, 1] for all dimensions."""

    def __init__(self, data: TensorType) -> None:
        super().__init__()
        self._min = None
        self._delta = None
        self._set_parameters(data)

    def _set_parameters(self, data: TensorType) -> TensorType:
        self._min = tf.math.reduce_min(data, axis=0)
        max_data = tf.math.reduce_max(data, axis=0)
        self._delta = max_data - self._min

    def transform(self, data: TensorType) -> TensorType:
        return (data - self._min) / self._delta

    def inverse_transform(self, data: TensorType) -> TensorType:
        return data * self._delta + self._min

    def transform_covariance(self, covariance: TensorType) -> TensorType:
        return covariance / self._delta ** 2

    def inverse_transform_covariance(self, covariance: TensorType) -> TensorType:
        return covariance * self._delta ** 2
