from abc import ABC, abstractmethod
from typing import Any, Tuple, Union

import tensorflow as tf

from trieste.data import Dataset
from trieste.types import TensorType

from .interfaces import TrainableProbabilisticModel


class DataTransformer(ABC):
    """A data transformer."""

    def __init__(self, data: TensorType) -> None:
        """Initialize object and set parameters of the transform.
        
        :param data: The unnormalized data.
        """
        super().__init__()
        self.set_parameters(data)

    @abstractmethod
    def set_parameters(self, data: TensorType) -> None:
        """Set parameters of the transformation either explicitly by passing them as keyword
        arguments, or based on the unnormalized data.

        :param data: The unnormalized data. If this argument is used is then the transform's
        parameters will be set from the data and should not be passed explicitly.
        :param transform_parameters: Keyword arguments for explicitly setting the transform's
        parameters. If these are used then the `data` keyword argument should be `None`.
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
        """[Optionally] transform the variance of the unnormalised data to the variance of the
        normalized data."""
        return variance

    def inverse_transform_variance(self, variance: TensorType) -> TensorType:
        """[Optionally] transform the variance of the normalised data to the variance of the
        unnormalized data."""
        return variance


class IdentityTransformer(DataTransformer):
    """Passes data through, unchanged."""

    def set_parameters(self, data: TensorType) -> None:
        """The IdentityTransform has no parameters."""
        pass

    def transform(self, data: TensorType) -> TensorType:
        """Return data, unchanged.
        
        :param data: Data to be passed through.
        """
        return data

    def inverse_transform(self, data: TensorType) -> TensorType:
        """Return data, unchanged.
        
        :param data: Data to be passed through.
        """
        return data


class StandardTransformer(DataTransformer):
    """Transforms data by subtracting mean and dividing by standard deviation."""

    def __init__(self, data: TensorType) -> None:
        """Initialize a transformer that performs "Standardisation" i.e. this transform will
        normalize to zero mean and unit variance.

        :param data: The unnormalized data [N, D].
        """
        super().__init__(data)

    def set_parameters(self, data: TensorType) -> None:
        """Set parameters of the transform; the mean (self._mean) and standard devation (self._std)
        of the data along each dimension.

        :param data: The unnormalized data [N, D].
        """
        self._mean = tf.squeeze(tf.math.reduce_mean(data, axis=0))
        self._std = tf.squeeze(tf.math.reduce_std(data, axis=0))

    def transform(self, data: TensorType) -> TensorType:
        """Normalize data.
        
        :param data: The unnormalized data. Shape must be [..., D] if transform operates over
        multiple dimensions else any shape.
        :return: The normalized data. Same shape as `data`.
        """
        return (data - self._mean) / self._std

    def inverse_transform(self, data: TensorType) -> TensorType:
        """Transform normalized data to unnormalized space.
        
        :param data: The normalized data. Shape must be [..., D] if transform operates over
        multiple dimensions else any shape.
        :return: The unnormalized data. Same shape as `data`.
        """
        return data * self._std + self._mean

    def transform_variance(self, variance: TensorType) -> TensorType:
        """Transform the variance of unnormalized data to the variance of normalized data.
        
        :param variance: The variance of the unnormalized data. Shape must be [..., D] if transform
        operates over multiple dimensions else any shape.
        :return: The variance of the normalized data. Same shape as `data`.
        """
        return variance / self._std ** 2

    def inverse_transform_variance(self, variance: TensorType) -> TensorType:
        """Transform the variance of normalized data to the variance of unnormalized data.

        :param variance: The variance of the unnormalized data. Shape must be [..., D] if transform
        operates over multiple dimensions else any shape.
        :return: The variance of the normalized data. Same shape as `data`.
        """
        return variance * self._std ** 2


class MinMaxTransformer(DataTransformer):
    """Transforms data to the unit cube, i.e. to the interval [0, 1], for all dimensions."""

    def __init__(self, data: TensorType) -> None:
        """Initialize a transformer that normalizes data to the unit cube.

        :param data: The unnormalized data [N, D].
        """
        super().__init__(data)

    def set_parameters(self, data: TensorType) -> None:
        """Set parameters of the transform; the minimum (self._min) and the difference between the
        maximum and the minimum (self._delta) of the data along each dimension.

        :param data: The unnormalized data [N, D].
        """
        self._min = tf.squeeze(tf.math.reduce_min(data, axis=0))
        max_data = tf.squeeze(tf.math.reduce_max(data, axis=0))
        self._delta = max_data - self._min

    def transform(self, data: TensorType) -> TensorType:
        """Normalize data.

        :param data: The unnormalized data. Shape must be [..., D] if transform operates over
        multiple dimensions else any shape.
        :return: The normalized data. Same shape as `data`.
        """
        return (data - self._min) / self._delta

    def inverse_transform(self, data: TensorType) -> TensorType:
        """Transform normalized data to unnormalized space.

        :param data: The normalized data. Shape [..., D] if transform operates over multiple
        dimensions else any shape.
        :return: The unnormalized data. Same shape as `data`.
        """
        return data * self._delta + self._min

    def transform_variance(self, variance: TensorType) -> TensorType:
        """Transform the variance of unnormalized data to the variance of normalized data.

        :param variance: The variance of the unnormalized data. Shape [..., D] if transform
        operates over multiple dimensions else any shape.
        :return: The variance of the normalized data. Same shape as `data`.
        """
        return variance / self._delta ** 2

    def inverse_transform_variance(self, variance: TensorType) -> TensorType:
        """Transform the variance of normalized data to the variance of unnormalized data.

        :param variance: The variance of the unnormalized data. Shape [..., D] if transform
        operates over multiple dimensions else any shape.
        :return: The variance of the normalized data. Same shape as `data`.
        """
        return variance * self._delta ** 2


class DataTransformModelWrapper(TrainableProbabilisticModel):
    """
    A base class that handles data transformations for :class: `TrainableProbabilisticModel`s.

    Usage requires creating a new class that inherits from this class and the desired model, e.g.
    ```
    class NormalizedModel(DataTransformWrapper, GaussianProcessRegression):
        pass
    ```
    Ensure that this class is the first class inherited from.

    To update model hyperparameters on each iteration, pass `update_parameters=True` into the
    constructor. It is likely that the user will require a bespoke implementation to handle this
    with their chosen model. For this, they will typically want to subclass the following methods:
        _initialize_model_parameters
        _update_model_and_normalization_parameters
    """

    def __init__(
        self,
        *model_args: Any,
        dataset: Union[Dataset, None] = None,
        query_point_transformer: Union[DataTransformer, None] = None,
        observation_transformer: Union[DataTransformer, None] = None,
        update_parameters: bool = False,
        **model_kwargs: Any
    ) -> None:
        """Construct the wrapped model.

        :param dataset: The unnormalized dataset.
        :param model_args: Positional arguments to be passed into the wrapped model constructor.
        :param query_point_transformer: Transformer for query points.
        :param observation_transformer: Transformer for observations.
        :param update_parameters: Whether to update the normalization and model parameters at each
            iteration of the Bayesian optimization loop. If `True` then the methods
            `_update_model_hyperparameters` and `_update_hyperparameter_priors` should be defined.
        :param model_kwargs: Keyword arguments to be passed into the wrapped model constructor.
        """
        super().__init__(*model_args, **model_kwargs)
        if dataset is None:
            raise TypeError('Initial dataset must be passed.')
        self._query_point_transformer = (
            query_point_transformer if query_point_transformer is not None
            else IdentityTransformer(dataset.query_points)
        )
        self._observation_transformer = (
            observation_transformer if observation_transformer is not None
            else IdentityTransformer(dataset.observations)
        )
        super().update(self._transform_dataset(dataset))

        self._update_parameters = update_parameters
        if self._update_parameters:
            self._initialize_model_parameters()

    def _transform_dataset(self, dataset: Dataset) -> Dataset:
        """Normalize dataset.

        :param dataset: The unnormalized dataset.
        :return: The normalized datset.
        """
        return Dataset(
            query_points=self._query_point_transformer.transform(dataset.query_points),
            observations=self._observation_transformer.transform(dataset.observations)
        )

    def _update_normalization_parameters(self, dataset: Dataset) -> None:
        """Update normalization parameters for the new dataset.

        :param dataset: New, unnormalized, dataset.
        """
        self._query_point_transformer.set_parameters(dataset.query_points)
        self._observation_transformer.set_parameters(dataset.observations)

    def _initialize_model_parameters(self) -> None:
        """Update initial model hyperparameters by transforming into normalized space."""
        pass

    def _update_model_and_normalization_parameters(self, dataset: Dataset) -> None:
        """Update the model and normalization parameters based on the new dataset.

        :param dataset: New, unnormalized, dataset.
        """
        self._update_normalization_parameters(dataset)

    def _predict(
        self, query_points: TensorType, predict_type: str = '', *args: Any, **kwargs: Any
    ) -> Tuple[TensorType, TensorType]:
        """For wrapping the model's prediction methods to feed in normalized query points and
        denormalize the output.

        :param query_points: The unnormalized query points [..., D].
        :param predict_type: The type of prediction; options are '', '_joint', or 'y'.
        :param args: Arguments to pass to the model's predict function.
        :param kwargs: Keyword arguments to pass to the model's predict function.
        :return: Denormalized predicted mean and denormalized predicted variance or covariance.
        """
        assert (
            predict_type == '' or '_joint' or '_y', f'predict_type {predict_type} not supported.'
        )
        transformed_query_points = self._query_point_transformer.transform(query_points)
        predict_function = getattr(super(), f'predict{predict_type}')
        mean, covariance = predict_function(transformed_query_points, *args, **kwargs)
        return (
            self._observation_transformer.inverse_transform(mean),
            self._observation_transformer.inverse_transform_variance(covariance)
        )

    def predict(
        self, query_points: TensorType, *args: Any, **kwargs: Any
    ) -> Tuple[TensorType, TensorType]:
        """Wrap model's `predict` method to pass in a normalized dataset and return unnormalized
        outputs.
        
        :param query_points: The unnormalized query points [..., D].
        :param args: Arguments to pass to the model's `predict` function.
        :param kwargs: Keyword arguments to pass to the model's `predict` function.
        :return: Denormalized predicted mean and denormalized predicted variance or covariance.
        """
        return self._predict(query_points, *args, **kwargs)

    def predict_joint(
        self, query_points: TensorType, *args: Any, **kwargs: Any
    ) -> Tuple[TensorType, TensorType]:
        """Wrap model's `predict_joint` method to pass in a normalized dataset and return
        unnormalized outputs.
        
        :param query_points: The unnormalized query points [..., D].
        :param args: Arguments to pass to the model's `predict_joint` function.
        :param kwargs: Keyword arguments to pass to the model's `predict_joint` function.
        :return: Denormalized predicted mean and denormalized predicted variance or covariance.
        """
        return self._predict(query_points, *args, predict_type='_joint', **kwargs)

    def sample(
        self, query_points: TensorType, num_samples: int, *args: Any, **kwargs: Any
    ) -> TensorType:
        """Wrap the model's `sample` method to normalize the inputs before sampling, and
        unnormalize the outputs before returning.

        :param query_points: Unnormalized query points [..., D].
        :param num_samples: Number of samples required, N.
        :param args: Additional arguments to pass to the model's `sample` function.
        :param kwargs: Keyword arguments to pass to the model's `sample` function.
        :return: Unnormalized samples [N, ..., D]
        """
        transformed_query_points = self._query_point_transformer.transform(query_points)
        samples = super().sample(transformed_query_points, *args, **kwargs)
        return self._observation_transformer.inverse_transform(samples)

    def predict_y(
        self, query_points: TensorType, *args: Any, **kwargs: Any
    ) -> Tuple[TensorType, TensorType]:
        """Wrap model's `predict_y` method to pass in a normalized dataset and return unnormalized
        outputs.

        :param query_points: The unnormalized query points [..., D].
        :param args: Arguments to pass to the model's `predict_y` function.
        :param kwargs: Keyword arguments to pass to the model's `predict_y` function.
        :return: Denormalized predicted mean and denormalized predicted variance or covariance.
        """
        return self._predict(query_points, *args, predict_type='_y', **kwargs)

    def update(self, dataset: Dataset, *args: Any, **kwargs: Any) -> None:
        """Wrap the model's `update` method to pass in a normalized dataset. Optionally update 
        normalization and model parameters.

        :param dataset: Unnormalized dataset.
        :param args: Arguments to pass to the model's `update` function.
        :param kwargs: Keyword arguments to pass to the model's `update` function.
        """
        if self._update_parameters:
            self._update_model_and_normalization_parameters(dataset)
        transformed_dataset = self._transform_dataset(dataset)
        super().update(transformed_dataset, *args, **kwargs)  # Will update data only.

    def optimize(self, dataset: Dataset, *args: Any, **kwargs: Any) -> None:
        """Wrap the model's `optimize` method to pass in a normalized dataset.

        :param dataset: Unnormalized dataset.
        :param args: Arguments to pass to the model's `optimize` function.
        :param kwargs: Keyword arguments to pass to the model's `optimize` function.
        """
        transformed_dataset = self._transform_dataset(dataset)
        super().optimize(transformed_dataset, *args, **kwargs)
