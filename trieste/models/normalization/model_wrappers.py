from typing import Mapping, Any, Tuple
import tensorflow as tf
from gpflow import default_float
from gpflow.utilities.traversal import read_values, multiple_assign

from .transformers import DataTransformer
from ..interfaces import TrainableProbabilisticModel
from ...data import Dataset
from ...types import S, TensorType


class DataTransformWrapper(TrainableProbabilisticModel):
    """
    A base class that handles data transformations for :class: `TrainableProbabilisticModel`s.

    Usage requires creating a new class that inherits from this class and the desired model, e.g.
        class NormalizedModel(DataTransformWrapper, GaussianProcessRegression):
            pass
    Ensure that this class is the first class inherited from.

    To update model hyperparameters on each iteration, pass `update_parameters=True` into the
    constructor. It is likely that the user will require a bespoke implementation to handle this
    with their chosen model. For this, they will typically want to subclass the following methods:
        _process_hyperparameter_dictionary
        _get_unnormalized_hyperparameter_priors
        _update_hyperparameter_priors
    """

    def __init__(
        self,
        dataset: Dataset,
        *model_args,
        query_point_transformer: DataTransformer,
        observation_transformer: DataTransformer,
        update_parameters: bool = False,
        **model_kwargs
    ) -> None:
        """Construct the wrapped model.
        
        :param dataset: The unnormalized dataset.
        :param model_args: Positional arguments to be passed into the wrapped model constructor.
        :param query_point_transformer: Transformer for query points.
        :param observation_transformer: Transformer for observations.
        :param update_parameters: Whether to update the normalization and model
            parameters at each iteration of the Bayesian optimization loop. If `True` then the methods
            `_update_model_hyperparameters` and `_update_hyperparameter_priors` should be defined.
        :param model_kwargs: Keyword arguments to be passed into the wrapped model constructor.
        """
        super().__init__(*model_args, **model_kwargs)
        self._query_point_transformer = query_point_transformer
        self._observation_transformer = observation_transformer
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

    def _process_hyperparameter_dictionary(self, hyperparameters, inverse_transform: bool = False):
        """Transform model hyperparameters based on data transforms.
        
        :param hyperparameters: The untransformed hyperparameters.
        :param inverse_transform: Whether to apply the forward transform (if False) or the inverse
            transform (if True).
        :returns: The transformed hyperparameters.
        """
        prefix = 'inverse_' if inverse_transform else ''
        processed_hyperparameters = {}
        for key, value in hyperparameters.items():
            tf_value = tf.constant(value, dtype=default_float())  # Ensure value is tf Tensor
            if key.endswith('mean_function.c'):
                transform = getattr(self._observation_transformer, f'{prefix}transform')
            elif key.endswith('variance') and not 'likelihood' in key:
                transform = getattr(self._observation_transformer, f'{prefix}transform_variance')
            elif key.endswith('lengthscales') or key.endswith('period'):
                transform = getattr(self._query_point_transformer, f'{prefix}transform')
            else:
                transform = lambda x: x
            processed_hyperparameters[key] = transform(tf_value)
        return processed_hyperparameters
    
    def _update_normalization_parameters(self, dataset: Dataset) -> None:
        """Update normalization parameters for the new dataset.
        
        :param dataset: New, unnormalized, dataset.
        """
        self._query_point_transformer.set_parameters(dataset.query_points)
        self._observation_transformer.set_parameters(dataset.observations)

    def _transform_and_assign_hyperparameters(
        self, hyperparameters: Mapping[str, TensorType]
    ) -> None:
        """Transform hyperparameters for normalized data, and assign to model.
        
        :param hyperparameters: Hyperparameters for unnormalized data.
        """
        normalized_hyperparameters = self._process_hyperparameter_dictionary(hyperparameters)
        multiple_assign(self._model, normalized_hyperparameters)
    
    def _get_unnormalised_hyperparameter_priors(self) -> Mapping[str, Any]:
        """Get hyperparameter priors from the model, and return distributions over hyperparameters
        for unnormalized data.
        """
        return None

    def _update_hyperparameter_priors(self, unnormalised_hyperparameter_priors: Mapping[str, Any]) -> None:
        """Update hyperparameter priors based upon the chosen normalizations.
        
        :param unnormalized_hyperparameter_priors: Priors over hyperparameters in unnormalized space.
        """
        pass

    def _initialize_model_parameters(self) -> None:
        """Update initial model hyperparameters by transforming into normalized space."""
        hyperparameters = read_values(self._model)
        hyperparameters = {k: tf.constant(v, dtype=default_float()) for k, v in hyperparameters.items()}
        self._transform_and_assign_hyperparameters(hyperparameters)

    def _update_model_and_normalization_parameters(self, dataset: Dataset) -> None:
        """Update the model and normalization parameters based on the new dataset.
        i.e. Denormalize using the old parameters, and renormalize using parameters set from
        the new dataset.
        
        :param dataset: New, unnormalized, dataset.
        """
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
        """For wrapping the model's prediction methods to feed in normalized query points and denormalize
        the output.

        :param query_points: The unnormalized query points [..., D].
        :param predict_type: The type of prediction; options are '', '_joint', or 'y'.
        """
        assert predict_type == '' or '_joint' or '_y', f'predict_type {predict_type} not supported.'
        transformed_query_points = self._query_point_transformer.transform(query_points)
        predict_function = getattr(super(), f'predict{predict_type}')
        mean, covariance = predict_function(transformed_query_points)
        return (
            self._observation_transformer.inverse_transform(mean),
            self._observation_transformer.inverse_transform_variance(covariance)
        )

    def predict(self, query_points: TensorType) -> Tuple[TensorType, TensorType]:
        """Wrap model's `predict` method to pass in a normalized dataset and return unnormalized outputs."""
        return self._predict(query_points)

    def predict_joint(self, query_points: TensorType) -> Tuple[TensorType, TensorType]:
        """Wrap model's `predict_joint` method to pass in a normalized dataset and return unnormalized outputs."""
        return self._predict(query_points, predict_type='_joint')

    def sample(self, query_points: TensorType, num_samples: int) -> TensorType:
        """Wrap the model's `sample` method to normalize the inputs before sampling, and unnormalize the outputs
        before returning.

        :param query_points: Unnormalized query points [..., D].
        :param num_samples: Number of samples required, N.
        :return: Unnormalized samples [N, ..., D]
        """
        transformed_query_points = self._query_point_transformer.transform(query_points)
        samples = super().sample(transformed_query_points, num_samples)
        return self._observation_transformer.inverse_transform(samples)

    def predict_y(self, query_points: TensorType) -> Tuple[TensorType, TensorType]:
        """Wrap model's `predict_y` method to pass in a normalized dataset and return unnormalized outputs."""
        return self._predict(query_points, predict_type='_y')

    def update(self, dataset: Dataset) -> None:
        """Wrap the model's `update` method to pass in a normalized dataset. Optionally update normalization
        and model parameters.
        
        :param dataset: Unnormalized dataset.
        """
        if self._update_parameters:
            self._update_model_and_normalization_parameters(dataset)
        transformed_dataset = self._transform_dataset(dataset)
        super().update(transformed_dataset)  # Will update data only.

    def optimize(self, dataset: Dataset) -> None:
        """Wrap the model's `optimize` method to pass in a normalized dataset.
        
        :param dataset: Unnormalized dataset.
        """
        transformed_dataset = self._transform_dataset(dataset)
        return super().optimize(transformed_dataset)
