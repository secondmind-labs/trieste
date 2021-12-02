from typing import Any

from trieste.data import Dataset
from trieste.types import TensorType
from trieste.utils import DEFAULTS

from ..transforms import DataTransformModelWrapper
from .models import GaussianProcessRegression, SparseVariational, VariationalGaussianProcess


class GaussianProcessRegressionDataTransformWrapper(
    DataTransformModelWrapper, GaussianProcessRegression
):
    """A wrapped `GaussianProcessRegression` model that handles data transformation. Inputs are
    transformed before passing to the superclass implementation. The outputs are inverse
    transformed before returning.
    """

    def covariance_between_points(
        self, query_points_1: TensorType, query_points_2: TensorType
    ) -> TensorType:
        """Wrap model's `covariance_between_points` method.

        :param query_points_1: Set of unnormalized query_points with shape [N, D].
        :param query_points_2: Set of unnormalized query_points with shape [M, D].
        :return: Denormalized covariance matrix with shape [N, M].
        """
        transformed_query_points_1 = self._query_point_transformer.transform(query_points_1)
        transformed_query_points_2 = self._query_point_transformer.transform(query_points_2)
        covariance = super().covariance_between_points(
            transformed_query_points_1, transformed_query_points_2
        )
        return self._observation_transformer.inverse_transform_variance(covariance)


class SparseVariationalDataTransformWrapper(DataTransformModelWrapper, SparseVariational):
    """A wrapped `SparseVariational` model that handles data transformation. Inputs are
    transformed before passing to the superclass implementation. The outputs are inverse
    transformed before returning.
    """

    pass


class VariationalGaussianProcessDataTransformWrapper(
    DataTransformModelWrapper, VariationalGaussianProcess
):
    """A wrapped `VariationalGaussianProcess` model that handles data transformation. Inputs are
    transformed before passing to the superclass implementation. The outputs are inverse
    transformed before returning.
    """

    def update(
        self, dataset: Dataset, *args: Any, jitter: float = DEFAULTS.JITTER, **kwargs: Any
    ) -> None:
        """Wraps the `update` method so that jitter is transformed to normalised space.

        :param dataset: The unnormalised dataset.
        :param args: Positional arguments to pass through to the superclass implementation.
        :param jitter: Jitter specified for the normalised space. Used for stabilizing the Cholesky
            decomposition of the covariance matrix.
        """
        transformed_jitter = self._observation_transformer.transform_variance(jitter)
        # Standard DataTransformModelWrapper update method will take care of transforming dataset
        # and then calling the VariationalGaussianProcess update method.
        return super().update(dataset, *args, jitter=transformed_jitter, **kwargs)
