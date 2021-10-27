from .interfaces import TrainableProbabilisticModel
from ..data import Dataset
from ..types import S, TensorType


class NormalizationMixin(TrainableProbabilisticModel):
    """
    A base class for mixins that handles data transformations for
    :class: `TrainableProbabilisticModel`s.

    Usage requires subclassing this class and creating a new class that inherits from the mixin
    subclass and the desired model, i.e.
        class NormalizedModel(SubNormalizationMixin, SubTrainableProbabilisticModel):
            pass
    Ensure that the mixin is the first class inherited from.

    A user will typically want to subclass the following methods:
        _transform_query_points
        _transform_observations
        _inverse_transform_query_points
        _inverse_transform_observations
        _inverse_transform_covariance
        _update_model_and_normalization_parameters
    """
    def __init__(self, **normalization_parameters) -> None:
        super().__init__()
        self.__dict__.update(normalization_parameters)

    def _transform_query_points(self, query_points: TensorType) -> TensorType:
        return query_points

    def _transform_observations(self, observations: TensorType) -> TensorType:
        return observations

    def _inverse_transform_query_points(self, query_points: TensorType) -> TensorType:
        return query_points

    def _inverse_transform_observations(self, observations: TensorType) -> TensorType:
        return observations

    def _inverse_transform_covariance(self, covariance: TensorType) -> TensorType:
        return covariance

    def _transform_dataset(self, dataset: Dataset) -> Dataset:
        return Dataset(
            query_points=self._transform_query_points(dataset.query_points),
            observations=self._transform_observations(dataset.observations)
        )

    def _update_model_and_normalization_parameters(self, dataset: Dataset) -> None:
        pass

    def predict(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        transformed_query_points = self._transform_query_points(query_points)
        mean, covariance = super().predict(transformed_query_points)
        return (
            self._inverse_transform_observations(mean),
            self._inverse_transform_covariance(covariance)
        )

    def sample(self, query_points: TensorType, num_samples: int) -> TensorType:
        transformed_query_points = self._transform_query_points(query_points)
        samples = super().sample(transformed_query_points, num_samples)
        return self._inverse_transform_observations(samples)

    def update(self, dataset: Dataset) -> None:
        self._update_model_and_normalization_parameters(dataset)
        transformed_dataset = self._transform_dataset(dataset)
        super().update(transformed_dataset)  # Will update data only.

    def optimize(self, dataset: Dataset) -> None:
        transformed_dataset = self._transform_dataset(dataset)
        return super().optimize(transformed_dataset)
