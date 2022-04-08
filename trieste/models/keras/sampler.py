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

import numpy as np
import tensorflow as tf
from tensorflow_probability.python.distributions import MultivariateNormalTriL

from ...types import TensorType
from ...utils import flatten_leading_dims
from ..interfaces import (
    EnsembleModel,
    TrajectoryFunction,
    TrajectoryFunctionClass,
    TrajectorySampler,
)


class EnsembleTrajectorySampler(TrajectorySampler[EnsembleModel]):
    """
    This class builds functions that approximate a trajectory by randomly choosing a network from
    the ensemble and using its predicted means as a trajectory.
    """

    def __init__(self, model: EnsembleModel, use_samples: bool = False, batch_size: int = 1):
        """
        :param model: The ensemble model to sample from.
        :param use_samples:
        :param batch_size:
        """
        if not isinstance(model, EnsembleModel):
            raise NotImplementedError(
                f"EnsembleTrajectorySampler only works with EnsembleModel models, that support "
                f"ensemble_size, sample_index, predict_ensemble and sample_ensemble methods; "
                f"received {model.__repr__()}"
            )

        super().__init__(model)

        self._model = model
        self._use_samples = use_samples
        self._batch_size = batch_size

    def __repr__(self) -> str:
        """"""
        return f"{self.__class__.__name__}({self._model!r}"

    def get_trajectory(self) -> TrajectoryFunction:
        """
        Generate an approximate function draw (trajectory) by randomly choosing a network from
        the ensemble and using its predicted means as a trajectory.

        :return: A trajectory function representing an approximate trajectory
            from the model, taking an input of shape `[N, 1, D]` and returning shape `[N, 1]`.
        """
        return ensemble_trajectory(self._model, self._use_samples, self._batch_size)

    def resample_trajectory(self, trajectory: TrajectoryFunction) -> TrajectoryFunction:
        """
        Efficiently resample a :const:`TrajectoryFunction` in-place to avoid function retracing
        with every new sample.

        :param trajectory: The trajectory function to be resampled.
        :return: The new resampled trajectory function.
        """
        tf.debugging.Assert(isinstance(trajectory, ensemble_trajectory), [])
        trajectory.resample()  # type: ignore
        return trajectory


class ensemble_trajectory(TrajectoryFunctionClass):
    """
    Generate an approximate function draw (trajectory) by randomly choosing a batch B of
    networks from the ensemble and using their predicted means as trajectories.
    """

    def __init__(self, model: EnsembleModel, use_samples: bool, batch_size: int):
        """
        :param model: The model of the objective function.
        :param use_samples:
        :param batch_size:
        """
        self._model = model
        self._use_samples = use_samples
        self._batch_size = batch_size
        self._indices = [0 for _ in range(batch_size)]
        if self._use_samples:
            self._seeds = tf.Variable(tf.ones([batch_size, 2], dtype=tf.int32), trainable=False)
        self.resample()

    @tf.function
    def __call__(self, x: TensorType) -> TensorType:  # [N, B, d] -> [N, B]
        """
        Call trajectory function. Note that we are flattening the batch dimension and
        doing a forward pass with each network in the ensemble with the whole batch. This is
        somewhat wasteful, but is necessary given the underlying ``KerasEnsemble`` network
        model. Also, if same networks are used in multiple batch elements due to sampling
        with replacement it is less wasteful.
        """
        tf.debugging.assert_equal(
            tf.shape(x)[-2],
            self._batch_size,
            message=f"""
            This trajectory only supports batch sizes of {self._batch_size}.
            If you wish to change the batch size you must get a new trajectory
            by calling the get_trajectory method of the trajectory sampler.
            """,
        )

        # TOFIX: we're currently assuming model is a DeepEnsemble, not any EnsembleModel
        flat_x, unflatten = flatten_leading_dims(x)  # [N*B, d]
        x_transformed: dict[str, TensorType] = self._model.prepare_query_points(flat_x)
        ensemble_distributions: tuple[MultivariateNormalTriL, ...] = self._model.model(
            x_transformed
        )

        predictions = []
        if self._use_samples:
            for b, seed in zip(range(self._batch_size), tf.unstack(self._seeds)):
                predictions.append(ensemble_distributions[self._indices[b]].sample(seed=seed))
        else:
            for b in range(self._batch_size):
                predictions.append(ensemble_distributions[self._indices[b]].mean())

        tensor_predictions = tf.squeeze(
            tf.convert_to_tensor([unflatten(p) for p in predictions]), axis=-1
        )  # [B, N, B]

        # here we select simultaneously networks and batch dimension according to batch indices
        # this is needed because we compute a whole batch with each network
        batch_index = tf.range(0, self._batch_size, 1)
        indices = tf.stack([batch_index, batch_index], axis=1)
        batch_predictions = tf.gather_nd(
            tf.transpose(tensor_predictions, perm=[0, 2, 1]), indices
        )  # [B,N]

        return tf.transpose(batch_predictions, perm=[1, 0])  # [N, B]

    def resample(self) -> None:
        """
        Efficiently resample network indices, and optionally quantiles, in-place without retracing.
        """
        for i in range(self._batch_size):
            self._indices[i] = np.random.randint(self._model.ensemble_size)
        if self._use_samples:
            self._seeds.assign(
                tf.random.uniform(
                    shape=(self._batch_size, 2), minval=1, maxval=999999999, dtype=tf.int32
                )
            )
