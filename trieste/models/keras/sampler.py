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

"""
This module is the home of the sampling functionality required by some
of the Trieste's Keras model wrappers.
"""

from __future__ import annotations

import tensorflow as tf

from ...types import TensorType
from ...utils import flatten_leading_dims
from ..interfaces import TrajectoryFunction, TrajectoryFunctionClass, TrajectorySampler
from .interface import DeepEnsembleModel
from .utils import sample_model_index


class DeepEnsembleTrajectorySampler(TrajectorySampler[DeepEnsembleModel]):
    """
    This class builds functions that approximate a trajectory by randomly choosing a network from
    the ensemble and using its predicted means as a trajectory. Currently we do not
    support models with multiple outputs.

    Option `diversify` can be used to increase the diversity in case of optimizing very large
    batches of trajectories. In this case the sampler will use quantiles from distributions rather
    than means as trajectories. Quantiles are sampled uniformly from a unit interval. When batch
    size is larger than the ensemble size, multiple quantiles will be used with the same network.
    """

    def __init__(self, model: DeepEnsembleModel, diversify: bool = False):
        """
        :param model: The ensemble model to sample from.
        :param diversify: Whether to use quantiles from final probabilistic layer as
            trajectories (`False` by default). See class docstring for details.
        """
        if not isinstance(model, DeepEnsembleModel):
            raise NotImplementedError(
                f"EnsembleTrajectorySampler only works with DeepEnsembleModel models, that support "
                f"ensemble_size and ensemble_distributions methods; "
                f"received {model.__repr__()}"
            )

        super().__init__(model)

        self._model = model
        self._diversify = diversify

    def __repr__(self) -> str:
        """"""
        return f"{self.__class__.__name__}({self._model!r}"

    def get_trajectory(self) -> TrajectoryFunction:
        """
        Generate an approximate function draw (trajectory) from the ensemble.

        :return: A trajectory function representing an approximate trajectory
            from the model, taking an input of shape `[N, 1, D]` and returning shape `[N, 1]`.
        """
        return deep_ensemble_trajectory(self._model, self._diversify)

    def update_trajectory(self, trajectory: TrajectoryFunction) -> TrajectoryFunction:
        """
        Update a :const:`TrajectoryFunction` to reflect an update in its
        underlying :class:`DeepEnsembleModel` and resample accordingly.

        Here we rely on the underlying models being updated and we only resample the trajectory.

        :param trajectory: The trajectory function to be resampled.
        :return: The new trajectory function updated for a new model
        """
        tf.debugging.Assert(isinstance(trajectory, deep_ensemble_trajectory), [])
        trajectory.resample()  # type: ignore
        return trajectory

    def resample_trajectory(self, trajectory: TrajectoryFunction) -> TrajectoryFunction:
        """
        Efficiently resample a :const:`TrajectoryFunction` in-place to avoid function retracing
        with every new sample.

        :param trajectory: The trajectory function to be resampled.
        :return: The new resampled trajectory function.
        """
        tf.debugging.Assert(isinstance(trajectory, deep_ensemble_trajectory), [])
        trajectory.resample()  # type: ignore
        return trajectory


class deep_ensemble_trajectory(TrajectoryFunctionClass):
    """
    Generate an approximate function draw (trajectory) by randomly choosing a batch B of
    networks from the ensemble and using their predicted means as trajectories.

    Option `diversify` can be used to increase the diversity in case of optimizing very large
    batches of trajectories. It is experimental feature and hence should be used with caution.

    Models with multiple outputs are not supported.
    """

    def __init__(self, model: DeepEnsembleModel, diversify: bool):
        """
        :param model: The model of the objective function.
        :param diversify: Whether to use samples from final probabilistic layer as trajectories
            or mean predictions.
        """
        self._model = model
        self._diversify = diversify
        self._ensemble_size = self._model.ensemble_size

        self._initialized = tf.Variable(False, trainable=False)
        self._batch_size = tf.Variable(0, dtype=tf.int32, trainable=False)
        self._indices = tf.Variable(tf.zeros([0], dtype=tf.int32), shape=[None], trainable=False)

        if self._diversify:
            self._sample_size = tf.Variable(0, dtype=tf.int32, trainable=False)
            self._seeds = tf.Variable(
                tf.zeros([0, 0], dtype=tf.float32), shape=[None, None], trainable=False
            )

    @tf.function
    def __call__(self, x: TensorType) -> TensorType:  # [N, B, d] -> [N, B]
        """
        Call trajectory function. Note that we are flattening the batch dimension and
        doing a forward pass with each network in the ensemble with the whole batch. This is
        somewhat wasteful, but is necessary given the underlying ``KerasEnsemble`` network
        model.
        """
        if not self._initialized:  # work out desired batch size from input
            self._batch_size.assign(tf.shape(x)[-2])  # B
            if self._diversify:
                self._sample_size.assign(-(-self._batch_size // self._ensemble_size))  # B/E
            self.resample()  # sample network indices/seeds
            self._initialized.assign(True)

        tf.debugging.assert_equal(
            tf.shape(x)[-2],
            self._batch_size,
            message=f"""
            This trajectory only supports batch sizes of {self._batch_size}.
            If you wish to change the batch size you must get a new trajectory
            by calling the get_trajectory method of the trajectory sampler.
            """,
        )
        flat_x, unflatten = flatten_leading_dims(x)  # [N*B, d]
        ensemble_distributions = self._model.ensemble_distributions(flat_x)

        if self._diversify:
            samples = tf.convert_to_tensor(
                [
                    tf.map_fn(dist.quantile, self._seeds[i])
                    for i, dist in enumerate(ensemble_distributions)
                ]
            )  # [E, B/E, N*B, 1]
            flattened_samples = tf.reshape(
                samples, [self._ensemble_size * self._sample_size, *samples.shape[2:]]
            )  # [E*B/E, N*B, 1]
            predictions = tf.gather(flattened_samples, self._indices)  # [B, N*B, 1]
        else:
            predicted_means = tf.convert_to_tensor([dist.mean() for dist in ensemble_distributions])
            predictions = tf.gather(predicted_means, self._indices)  # [B, N*B, 1]

        tensor_predictions = tf.squeeze(tf.map_fn(unflatten, predictions), axis=-1)  # [B, N, B]

        # here we select simultaneously networks and batch dimension according to batch indices
        # this is needed because we compute a whole batch with each network
        batch_index = tf.range(self._batch_size)
        indices = tf.stack([batch_index, batch_index], axis=1)
        batch_predictions = tf.gather_nd(
            tf.transpose(tensor_predictions, perm=[0, 2, 1]), indices
        )  # [B,N]

        return tf.transpose(batch_predictions, perm=[1, 0])  # [N, B]

    def resample(self) -> None:
        """
        Efficiently resample network indices in-place, without retracing.
        """
        if self._diversify:
            indices = tf.random.shuffle(tf.range(self._ensemble_size * self._sample_size))
            self._indices.assign(indices[: self._batch_size])  # [B]
            self._seeds.assign(
                tf.random.uniform(
                    shape=(self._ensemble_size, self._sample_size),
                    minval=0.000001,
                    maxval=0.999999,
                    dtype=tf.float32,
                )
            )  # [E, B/E]
        else:
            self._indices.assign(sample_model_index(self._ensemble_size, self._batch_size))  # [B]
