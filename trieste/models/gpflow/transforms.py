from typing import Any, Tuple

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

    def _conditional_predict(
        self, query_points: TensorType, additional_data: Dataset, predict_type: str = "f"
    ) -> Tuple[TensorType, TensorType]:
        """Wrap conditional predict methods to normalize the query points and additional data, and
        denormalize the posterior.

        :param query_points: The unnormalized query_points [M, D].
        :param addtional_data: The unnormalized additional dataset with query_points with shape
            [..., N, D] and observations with shape [..., N, L].
        :return: Denormalized predictive mean [..., M, L] and denormalized predictive variance
            [..., M, L] or covariance [..., L, M, M].
        """
        assert predict_type == "f" or "joint" or "y", f"predict_type {predict_type} not supported"
        transformed_query_points = self._query_point_transformer.transform(query_points)
        transformed_additional_data = self._transform_dataset(additional_data)
        predict_function = getattr(super(), f"conditional_predict_{predict_type}")
        mean, covariance = predict_function(transformed_query_points, transformed_additional_data)
        return (
            self._observation_transformer.inverse_transform(mean),
            self._observation_transformer.inverse_transform_variance(covariance),
        )

    def conditional_predict_f(
        self, query_points: TensorType, additional_data: Dataset
    ) -> Tuple[TensorType, TensorType]:
        """Wrap model's `conditional_predict_f` method to normalize the query points and additional
        data, and denormalize the posterior.

        :param query_points: The unnormalized query_points [M, D].
        :param addtional_data: The unnormalized additional dataset with query_points with shape
            [..., N, D] and observations with shape [..., N, L].
        :return: Denormalized predictive mean [..., M, L] and denormalized predictive variance
            [..., M, L].
        """
        return self._conditional_predict(query_points, additional_data, predict_type="f")

    def conditional_predict_joint(
        self, query_points: TensorType, additional_data: Dataset
    ) -> Tuple[TensorType, TensorType]:
        """Wrap model's `conditional_predict_joint` method to normalize the query points and
        additional data, and denormalize the posterior.

        :param query_points: The unnormalized query_points [M, D].
        :param addtional_data: The unnormalized additional dataset with query_points with shape
            [..., N, D] and observations with shape [..., N, L].
        :return: Denormalized predictive mean [..., M, L] and denormalized predictive covariance
            [..., L, M, M].
        """
        return self._conditional_predict(query_points, additional_data, predict_type="joint")

    def conditional_predict_y(
        self, query_points: TensorType, additional_data: Dataset
    ) -> Tuple[TensorType, TensorType]:
        """Wrap model's `conditional_predict_f` method to normalize the query points and additional
        data, and denormalize the posterior.

        :param query_points: The unnormalized query_points [M, D].
        :param addtional_data: The unnormalized additional dataset with query_points with shape
            [..., N, D] and observations with shape [..., N, L].
        :return: Denormalized predictive mean [..., M, L] and denormalized predictive variance
            [..., M, L] or covariance [..., L, M, M].
        """
        return self._conditional_predict(query_points, additional_data, predict_type="y")

    def conditional_predict_f_sample(
        self, query_points: TensorType, additional_data: Dataset, num_samples: int
    ) -> TensorType:
        """Wrap model's `conditional_predict_f_sample` method to normalize the query points and
        additional data, and denormalize the samples.

        :param query_points: The unnormalized query_points [M, D].
        :param addtional_data: The unnormalized additional dataset with query_points with shape
            [..., N, D] and observations with shape [..., N, L].
        :return: Denormalized samples of f at query points, with shape [..., num_samples, M, L].
        """
        transformed_query_points = self._query_point_transformer.transform(query_points)
        transformed_additional_data = self._transform_dataset(additional_data)
        samples = super().conditional_predict_f_sample(
            transformed_query_points, transformed_additional_data, num_samples
        )
        return self._observation_transformer.inverse_transform(samples)


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
