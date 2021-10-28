from typing import Optional, Mapping, Any, Tuple
import types
import tensorflow as tf
from gpflow import default_float
from gpflow.utilities.traversal import read_values, multiple_assign

from .interfaces import TrainableProbabilisticModel
from ..data import Dataset
from ..types import S, TensorType


class NormalizationMixin(TrainableProbabilisticModel):
    """
    A base class for mixins that handles data transformations for
    :class: `TrainableProbabilisticModel`s.

    Usage requires subclassing this class and creating a new class that inherits from your mixin
    subclass and the desired model, i.e.
        class NormalizedModel(SubNormalizationMixin, SubTrainableProbabilisticModel):
            pass
    Ensure that the mixin is the first class inherited from.

    A user will typically want to subclass the following methods:
        _transform_query_points
        _transform_observations
        _transform_covariance
        _inverse_transform_query_points
        _inverse_transform_observations
        _inverse_transform_covariance
        _update_normalization_parameters
    """

    def __init__(
        self,
        dataset: Dataset,
        *args,
        normalization_parameters: Optional[Mapping[str, Any]] = None,
        **kwargs
    ) -> None:
        normalization_parameters = (
            {} if normalization_parameters is None else normalization_parameters
        )
        self.__dict__.update(normalization_parameters)
        super().__init__(*args, **kwargs)
        self._initialize_model_and_normalization_parameters(dataset)

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

    def _process_hyperparameter_dictionary(self, hyperparameters, inverse_transform: bool = False):
        prefix = '_inverse' if inverse_transform else ''
        processed_hyperparameters = {}
        for key, value in hyperparameters.items():
            tf_value = tf.constant(value, dtype=default_float())  # Ensure value is tf Tensor
            if key.endswith('mean_function.c'):
                transform = getattr(self, f'{prefix}_transform_observations')
            elif key.endswith('variance'):
                transform = getattr(self, f'{prefix}_transform_covariance')
            elif key.endswith('lengthscales') or key.endswith('period'):
                transform = getattr(self, f'{prefix}_transform_query_points')
            else:
                transform = lambda x: x
            processed_hyperparameters[key] = transform(tf_value)
        return processed_hyperparameters
    
    def _update_normalization_parameters(self, dataset: Dataset) -> None:
        pass

    def _transform_and_assign_hyperparameters(
        self, hyperparameters: Mapping[str, TensorType]
    ) -> None:
        normalized_hyperparameters = self._process_hyperparameter_dictionary(hyperparameters)
        multiple_assign(self._model, normalized_hyperparameters)
    
    def _get_unnormalised_hyperparameter_priors(self) -> Mapping[str, Any]:
        return None

    def _update_hyperparameter_priors(self, unnormalised_hyperparameter_priors: Mapping[str, Any]) -> None:
        pass

    def _initialize_model_and_normalization_parameters(self, dataset: Dataset) -> None:
        self._update_normalization_parameters(dataset)
        hyperparameters = read_values(self._model)
        hyperparameters = {k: tf.constant(v, dtype=default_float()) for k, v in hyperparameters.items()}
        self._transform_and_assign_hyperparameters(hyperparameters)

    def _update_model_and_normalization_parameters(self, dataset: Dataset) -> None:
        hyperparameters = read_values(self._model)
        unnormalized_hyperparameters = self._process_hyperparameter_dictionary(
            hyperparameters, inverse_transform=True
        )
        unnormalized_hyperparameter_priors = self._get_unnormalised_hyperparameter_priors()
        self._update_normalization_parameters(dataset)
        self._transform_and_assign_hyperparameters(unnormalized_hyperparameters)
        self._update_hyperparameter_priors(unnormalized_hyperparameter_priors)

    def _predict(
        self, query_points: TensorType, predict_type: str = ''
    ) -> Tuple[TensorType, TensorType]:
        transformed_query_points = self._transform_query_points(query_points)
        predict_function = getattr(super(), f'predict{predict_type}')
        mean, covariance = predict_function(transformed_query_points)
        return (
            self._inverse_transform_observations(mean),
            self._inverse_transform_covariance(covariance)
        )

    def predict(self, query_points: TensorType) -> Tuple[TensorType, TensorType]:
        return self._predict(query_points)

    def predict_joint(self, query_points: TensorType) -> Tuple[TensorType, TensorType]:
        return self._predict(query_points, predict_type='_joint')

    def sample(self, query_points: TensorType, num_samples: int) -> TensorType:
        transformed_query_points = self._transform_query_points(query_points)
        samples = super().sample(transformed_query_points, num_samples)
        return self._inverse_transform_observations(samples)

    def predict_y(self, query_points: TensorType) -> Tuple[TensorType, TensorType]:
        return self._predict(query_points, predict_type='_y')

    def update(self, dataset: Dataset) -> None:
        self._update_model_and_normalization_parameters(dataset)
        transformed_dataset = self._transform_dataset(dataset)
        super().update(transformed_dataset)  # Will update data only.

    def optimize(self, dataset: Dataset) -> None:
        transformed_dataset = self._transform_dataset(dataset)
        return super().optimize(transformed_dataset)


class StandardizationMixin(NormalizationMixin):
    """
    Performs standardization of observations.

    Ensures that the observations have zero mean and unit variance.
    """

    def _standardize(self, tensor: TensorType, mean: TensorType, std: TensorType) -> TensorType:
        return (tensor - mean) / std

    def _inverse_standardize(
        self, tensor: TensorType, mean: TensorType, std: TensorType
    ) -> TensorType:
        return tensor * std + mean

    def _transform_query_points(self, query_points: TensorType) -> TensorType:
        return self._standardize(query_points, self._query_points_mean, self._query_points_std)

    def _transform_observations(self, observations: TensorType) -> TensorType:
        return self._standardize(
            observations, self._observations_mean, self._observations_std
        )

    def _transform_covariance(self, covariance: TensorType) -> TensorType:
        return covariance / self._observations_std ** 2

    def _inverse_transform_query_points(self, query_points: TensorType) -> TensorType:
        return self._inverse_standardize(
            query_points, self._query_points_mean, self._query_points_std
        )

    def _inverse_transform_observations(self, observations: TensorType) -> TensorType:
        return self._inverse_standardize(
            observations, self._observations_mean, self._observations_std
        )

    def _inverse_transform_covariance(self, covariance: TensorType) -> TensorType:
        return covariance * self._observations_std ** 2

    def _update_normalization_parameters(self, dataset: Dataset):
        self._query_points_mean = tf.math.reduce_mean(dataset.query_points, axis=0)
        self._query_points_std  = tf.math.reduce_std(dataset.query_points, axis=0)
        self._observations_mean = tf.math.reduce_mean(dataset.observations)
        self._observations_std = tf.math.reduce_std(dataset.observations)
