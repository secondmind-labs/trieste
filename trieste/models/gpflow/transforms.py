from trieste.types import TensorType

from ..transforms import DataTransformModelWrapper
from .models import GaussianProcessRegression, SparseVariational, VariationalGaussianProcess


class GaussianProcessRegressionDataTransformWrapper(DataTransformModelWrapper):
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


class SparseVariationalwithDataTransform(DataTransformModelWrapper, SparseVariational):
    """A wrapped `SparseVariational` model that handles data transformation. Inputs are
    transformed before passing to the superclass implementation. The outputs are inverse
    transformed before returning.
    """

    pass


class VariationalGaussianProcesswithDataTransform(
    DataTransformModelWrapper, VariationalGaussianProcess
):
    """A wrapped `VariationalGaussianProcess` model that handles data transformation. Inputs are
    transformed before passing to the superclass implementation. The outputs are inverse
    transformed before returning.

    **Note**: The `update` method does not modify the `jitter` keyword argument. If this is
    desired, the user can achieve this by overloading the `update` method in a subclass.
    """

    pass
