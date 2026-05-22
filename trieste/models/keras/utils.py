# Copyright 2021 The Trieste Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Any, Callable, Optional, Union

import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.keras import tf_keras

from ...data import Dataset
from ...types import TensorType


def get_tensor_spec_from_data(dataset: Dataset) -> tuple[tf.TensorSpec, tf.TensorSpec]:
    r"""
    Extract tensor specifications for inputs and outputs of neural network models, based on the
    dataset. This utility faciliates constructing neural networks, providing the required
    dimensions for the input and the output of the network. For example

    >>> data = Dataset(
    ...     tf.constant([[0.1, 0.2], [0.3, 0.4]]),
    ...     tf.constant([[0.5], [0.7]])
    ... )
    >>> input_spec, output_spec = get_tensor_spec_from_data(data)
    >>> input_spec
    TensorSpec(shape=(2,), dtype=tf.float32, name='query_points')
    >>> output_spec
    TensorSpec(shape=(1,), dtype=tf.float32, name='observations')

    :param dataset: A dataset with ``query_points`` and ``observations`` tensors.
    :return: Tensor specification objects for the ``query_points`` and ``observations`` tensors.
    :raise ValueError: If the dataset is not an instance of :class:`~trieste.data.Dataset`.
    """
    if not isinstance(dataset, Dataset):
        raise ValueError(
            f"This function works only on trieste.data.Dataset objects, however got"
            f"{type(dataset)} which is incompatible."
        )
    input_tensor_spec = tf.TensorSpec(
        shape=(dataset.query_points.shape[1:]),
        dtype=dataset.query_points.dtype,
        name="query_points",
    )
    output_tensor_spec = tf.TensorSpec(
        shape=(dataset.observations.shape[1:]),
        dtype=dataset.observations.dtype,
        name="observations",
    )
    return input_tensor_spec, output_tensor_spec


def sample_with_replacement(dataset: Dataset) -> Dataset:
    """
    Create a new ``dataset`` with data sampled with replacement. This
    function is useful for creating bootstrap samples of data for training ensembles.

    :param dataset: The data that should be sampled.
    :return: A (new) ``dataset`` with sampled data.
    :raise ValueError (or InvalidArgumentError): If the dataset is not an instance of
        :class:`~trieste.data.Dataset` or it is empty.
    """
    if not isinstance(dataset, Dataset):
        raise ValueError(
            f"This function works only on trieste.data.Dataset objects, however got"
            f"{type(dataset)} which is incompatible."
        )
    tf.debugging.assert_positive(len(dataset), message="Dataset must not be empty.")

    n_rows = dataset.observations.shape[0]

    index_tensor = tf.random.uniform((n_rows,), maxval=n_rows, dtype=tf.dtypes.int32)

    observations = tf.gather(dataset.observations, index_tensor, axis=0)
    query_points = tf.gather(dataset.query_points, index_tensor, axis=0)

    return Dataset(query_points=query_points, observations=observations)


def sample_model_index(
    size: TensorType,
    num_samples: TensorType,
    seed: Optional[int] = None,
) -> TensorType:
    """
    Returns samples of indices of individual models in the ensemble.

    If ``num_samples`` is smaller or equal to ``size`` (i.e. the ensemble size) indices are sampled
    without replacement. When ``num_samples`` is larger than ``size`` then until ``size`` is reached
    we sample without replacement, while after that we sample with replacement. The rationale of
    this mixed scheme is that typically one wants to exhaust all networks and then resample them
    only if required.

    :param size: The maximum index, effectively the number of models in the ensemble.
    :param num_samples: The number of samples to take.
    :param seed: Optional RNG seed.
    :return: A tensor with indices.
    """
    shuffle_indices = tf.random.shuffle(tf.range(size), seed=seed)
    if num_samples > size:
        random_indices = tf.random.uniform(
            shape=(tf.cast(num_samples - size, tf.int32),),
            maxval=size,
            dtype=tf.int32,
            seed=seed,
        )
        indices = tf.concat([shuffle_indices, random_indices], 0)
    else:
        indices = shuffle_indices[:num_samples]

    return indices


def negative_log_likelihood(
    y_true: TensorType, y_pred: tfp.distributions.Distribution
) -> TensorType:
    """
    Maximum likelihood objective function for training neural networks.

    :param y_true: The output variable values.
    :param y_pred: The output layer of the model. It has to be a probabilistic neural network
        with a distribution as a final layer.
    :return: Negative log likelihood values.
    """
    return -y_pred.log_prob(y_true)


def aggregate_member_losses(
    loss_fn: Callable[[TensorType, Any], TensorType],
) -> Callable[[TensorType, Any], TensorType]:
    """
    Wrap a per-sample loss so the compiled scalar matches legacy multi-output Keras ensembles.

    Legacy models compile ``loss=[fn] * E``; Keras sums E batch-mean losses. Vectorized models
    have one output with shape ``[batch, E, ...]``; default Keras reduction averages over all
    axes, scaling gradients by about ``1/E``. This wrapper returns
    ``sum_m mean_batch(loss[..., m, ...])`` instead.

    :param loss_fn: Loss function, typically :func:`negative_log_likelihood`.
    :return: Loss function returning a scalar for vectorized single-output models.
    """

    def aggregated(y_true: TensorType, y_pred: Any) -> TensorType:
        values = loss_fn(y_true, y_pred)
        if len(values.shape) == 0:
            return values
        if len(values.shape) >= 2:
            return tf.reduce_sum(tf.reduce_mean(values, axis=0))
        return tf.reduce_mean(values, axis=0)

    return aggregated


def ensemble_negative_log_likelihood(
    y_true: TensorType, y_pred: tfp.distributions.Distribution
) -> TensorType:
    """
    Negative log-likelihood for vectorized ensembles (sum of per-member batch-mean NLL).

    :param y_true: Observations, shape ``[batch, E, ...]``.
    :param y_pred: Distribution with batch shape ``[batch, E, ...]``.
    :return: Scalar loss matching legacy ``loss=[negative_log_likelihood] * E`` compile.
    """
    return aggregate_member_losses(negative_log_likelihood)(y_true, y_pred)


_STRING_METRIC_CLASSES: dict[str, type[tf_keras.metrics.Metric]] = {
    "mse": tf_keras.metrics.MeanSquaredError,
    "mae": tf_keras.metrics.MeanAbsoluteError,
    "mape": tf_keras.metrics.MeanAbsolutePercentageError,
    "msle": tf_keras.metrics.MeanSquaredLogarithmicError,
}


def _metric_with_unique_name(metric: Any, name: str) -> tf_keras.metrics.Metric:
    if isinstance(metric, str):
        metric_class = _STRING_METRIC_CLASSES.get(metric)
        if metric_class is not None:
            return metric_class(name=name)
        resolved = tf_keras.metrics.get(metric)
        if isinstance(resolved, type) and issubclass(resolved, tf_keras.metrics.Metric):
            return resolved(name=name)
        return tf_keras.metrics.MeanMetricWrapper(resolved, name=name)
    if isinstance(metric, type) and issubclass(metric, tf_keras.metrics.Metric):
        return metric(name=name)
    if isinstance(metric, tf_keras.metrics.Metric):
        return type(metric).from_config({**metric.get_config(), "name": name})
    return tf_keras.metrics.MeanMetricWrapper(metric, name=name)


def _default_vectorized_metric_name(metric: Any) -> str:
    if isinstance(metric, str):
        return f"ensemble_{metric}"
    if isinstance(metric, tf_keras.metrics.Metric):
        return f"ensemble_{metric.name}"
    return "ensemble_metric"


def compile_metrics_for_ensemble(
    n_outputs: int,
    metrics: Optional[Union[list[Any], Any]],
    ensemble_size: Optional[int] = None,
) -> Optional[Union[list[tf_keras.metrics.Metric], tf_keras.metrics.Metric]]:
    """
    Build Keras compile metrics with unique names for vectorized and legacy ensemble layouts.

    Vectorized models use one output with shape ``[batch, E, ...]``; a single ``"mse"`` metric can
    collide with per-member metrics when ``steps_per_execution`` > 1. Legacy multi-output models
    need one uniquely named metric per output.

    :param n_outputs: Number of Keras model outputs.
    :param metrics: Metric(s) from :class:`~trieste.models.optimizer.KerasOptimizer`, or ``None``.
    :param ensemble_size: Ensemble size for legacy multi-output models (defaults to ``n_outputs``).
    :return: Metrics argument for :meth:`tf.keras.Model.compile`, or ``None``.
    """
    if metrics is None:
        return None

    if n_outputs == 1:
        if isinstance(metrics, (list, tuple)):
            if len(metrics) == 1:
                return [_metric_with_unique_name(metrics[0], _default_vectorized_metric_name(metrics[0]))]
            return [
                _metric_with_unique_name(
                    metric, f"{_default_vectorized_metric_name(metric)}_{index}"
                )
                for index, metric in enumerate(metrics)
            ]
        return [_metric_with_unique_name(metrics, _default_vectorized_metric_name(metrics))]

    n_members = ensemble_size if ensemble_size is not None else n_outputs
    if isinstance(metrics, (list, tuple)) and len(metrics) == n_members:
        return [
            _metric_with_unique_name(metric, f"{_metric_base_name(metric)}_{index}")
            for index, metric in enumerate(metrics)
        ]

    base = metrics[0] if isinstance(metrics, (list, tuple)) and len(metrics) == 1 else metrics
    return [_metric_with_unique_name(base, f"{_metric_base_name(base)}_{index}") for index in range(n_members)]


def _metric_base_name(metric: Any) -> str:
    if isinstance(metric, str):
        return metric
    if isinstance(metric, tf_keras.metrics.Metric):
        return metric.name
    return "metric"
