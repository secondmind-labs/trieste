# Copyright 2020 The Trieste Contributors
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

""" This module contains interface for scalarization functions and some example functions. """

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Mapping

import tensorflow as tf

from trieste.data import Dataset
from trieste.models.interfaces import (
    HasTrajectorySampler,
)
from trieste.types import Tag, TensorType


IdealSpecCallable = Callable[..., TensorType]
"""
Type alias for ideal spec callable type that takes ``models`` and ``datasets`` as arguments and
returns an ideal point with shape [K, 1].
"""


class Scalarizer(ABC):
    """
    This class provides an interface for scalarization functions. 
    It has two tasks: 1) scalarize the objectives, 2) sample the parameters that are used for
    diversifying the batch.

    There are many choices for sacalarization functions, for an overview of some number of them
    and empirical performance comparison, see https://arxiv.org/pdf/1904.05760.pdf
    """

    def __init__(
        self,
        batch_size: int,
        num_objectives: int,
        ideal_spec: Union[TensorType, IdealSpecCallable],
    ):
        """
        :param batch_size: Number of elements in the batch, B.
        :param num_objectives: Number of the objectives in the problem, K.
        :param ideal_spec: This argument is used to determine how the ideal point is
            calculated. If a Callable function specified, it is expected to take existing
            ``models`` and ``datasets`` as arguments and return an ideal point with shape [K, 1]. 
            If the Pareto front location is known, this arg can be used to specify a fixed ideal
            point in each active learning iteration, as a tensor with minimum values, [K, 1].
        """
        self._batch_size = batch_size
        self._num_objectives = num_objectives

        if callable(ideal_spec):
            self._ideal_spec: Union[TensorType, IdealSpecCallable] = ideal_spec
        else:
            self._ideal_spec = tf.convert_to_tensor(ideal_spec)

    @staticmethod
    def _infer_dtype(datasets: Mapping[Tag, Dataset]) -> tf.DType:
        return next(iter(datasets.values())).query_points.dtype

    def _get_ideal(
        self, models: Mapping[Tag, HasTrajectorySampler], datasets: Mapping[Tag, Dataset]
    ) -> TensorType:
        dtype = self._infer_dtype(datasets)
        if callable(self._ideal_spec):
            ideal = tf.cast(self._ideal_spec(models, datasets), dtype=dtype)
        else:
            ideal = tf.cast(self._ideal_spec, dtype=dtype)
        tf.debugging.assert_shapes([(ideal, (self._num_objectives, 1))])
        return ideal

    @abstractmethod
    def __call__(self, trajectories_at_x: TensorType) -> TensorType:
        """
        Perform the scalarization on objectives evaluated on some inputs ``trajectories_at_x``.

        :param trajectories_at_x: K objectives evaluated on N inputs, [K, N, B].
        :return: Scalarized objectives, [N, B].
        """

    @abstractmethod
    def prepare(
        self, models: Mapping[Tag, HasTrajectorySampler], datasets: Mapping[Tag, Dataset]
    ) -> None:
        """
        Generate all the internal variables on initialization. For example, weights in a linear
        weighted sum scalarization could be sampled.
        """

    @abstractmethod
    def update(
        self, models: Mapping[Tag, HasTrajectorySampler], datasets: Mapping[Tag, Dataset]
    ) -> None:
        """
        Update all the internal variables. Meant to be used between the bayesian optimization
        steps to change the state of the variables. For example, weights in a linear weighted
        sum scalarization could be resampled.
        """


class Chebyshev(Scalarizer):
    """
    A popular scalarization function, improvement over linear weighted sum in that it can deal
    with non-convex Pareto fronts. Along with weights, it has an ideal point parameter
    which acts as a reference point for computing the Chebyshev distance.
    In case we know the minimum of the Pareto front, we can set it to that. Otherwise
    a function can be used that would adaptively choose the ideal point based on the models or
    acquired data.

    By setting ``alpha`` to a value larger than zero we can switch to augmented Chebyshev
    scalarisation function which can help with avoiding weakly Pareto optimal solutions.

    See number 6 and 7 in https://arxiv.org/pdf/1904.05760.pdf for details, and for empirical
    comparison to other functions.
    """

    def __init__(
        self,
        batch_size: int,
        num_objectives: int,
        ideal_spec: Union[TensorType, IdealSpecCallable],
        alpha: float = 0.0,
    ):
        """
        :param batch_size: Number of elements in the batch, B.
        :param num_objectives: Number of the objectives in the problem, K.
        :param ideal_spec: This argument is used to determine how the ideal point is
            calculated. If a Callable function specified, it is expected to take existing
            ``models`` and ``datasets`` as arguments and return an ideal point with shape [K, 1]. 
            If the Pareto front location is known, this arg can be used to specify a fixed ideal
            point in each active learning iteration, as a tensor with minimum values, [K, 1].
        :param alpha: Parameter for augmented Chebyshev, set to zero by default. A good
            value for this parameter might be 0.0001, suggested in the reference above.
        """
        super().__init__(batch_size, num_objectives, ideal_spec)

        tf.debugging.assert_non_negative(alpha, f"alpha should be nonnegative but received {alpha}")
        self._alpha = alpha

    def __repr__(self) -> str:
        """"""
        if callable(self._ideal_spec):
            return (
                f"Chebyshev({self._batch_size!r},"
                f"{self._num_objectives!r},"
                f"{self._ideal_spec.__name__},"
                f"{self._alpha!r})"
            )
        else:
            return (
                f"Chebyshev({self._batch_size!r},"
                f"{self._num_objectives!r},"
                f"{self._ideal_spec!r},"
                f"{self._alpha!r})"
            )

    def __call__(self, trajectories_at_x: TensorType) -> TensorType:

        tf.debugging.assert_shapes(
            [
                (trajectories_at_x, (self._num_objectives, None, self._batch_size)),
                (self._weights, (self._num_objectives, self._batch_size)),
            ]
        )
        differences = tf.abs(trajectories_at_x - self._ideal[:, :, None])
        scalar = tf.reduce_max(
            self._weights[:, None, :] * differences, axis=0
        ) + self._alpha * tf.reduce_sum(differences, axis=0)
        tf.debugging.assert_shapes([(scalar, (None, self._batch_size))])

        return scalar

    def _sample_weights(self, dtype: tf.DType) -> TensorType:
        """
        Sample weights used for scalarizing the objectives.

        :return: Sampled weights [K, B].
        """
        weights = tf.random.normal([self._num_objectives, self._batch_size], dtype=dtype)
        return tf.abs(weights / tf.sqrt(tf.reduce_sum(weights ** 2, axis=0, keepdims=True)))

    def prepare(
        self, models: Mapping[Tag, HasTrajectorySampler], datasets: Mapping[Tag, Dataset]
    ) -> None:
        dtype = self._infer_dtype(datasets)
        self._weights = tf.Variable(  # pylint: disable=attribute-defined-outside-init
            self._sample_weights(dtype), trainable=False
        )
        self._ideal = tf.Variable(  # pylint: disable=attribute-defined-outside-init
            self._get_ideal(models, datasets), trainable=False
        )

    def update(
        self, models: Mapping[Tag, HasTrajectorySampler], datasets: Mapping[Tag, Dataset]
    ) -> None:
        dtype = self._infer_dtype(datasets)
        self._weights.assign(self._sample_weights(dtype))
        self._ideal.assign(self._get_ideal(models, datasets))
