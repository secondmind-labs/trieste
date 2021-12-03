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

from abc import ABC, abstractmethod
from typing import Callable

import tensorflow as tf

from ..types import TensorType
from ..utils import DEFAULTS
from .interfaces import ProbabilisticModel


class ReparametrizationSampler(ABC):
    r"""
    Thes samplers employ the *reparameterization trick* to draw samples from a
    :class:`ProbabilisticModel`\ 's predictive distribution  across a discrete set of
    points.
    """

    def __init__(self, sample_size: int, model: ProbabilisticModel):
        """
        :param sample_size: The desired number of samples.
        :param model: The model to sample from.
        :raise ValueError (or InvalidArgumentError): If ``sample_size`` is not positive.
        """
        tf.debugging.assert_positive(sample_size)

        self._sample_size = sample_size
        self._model = model
        self._initialized = tf.Variable(False)

    def __repr__(self) -> str:
        """"""
        return f"{self.__class__.__name__}({self._sample_size!r}, {self._model!r})"

    @abstractmethod
    def sample(self, at: TensorType, *, jitter: float = DEFAULTS.JITTER) -> TensorType:
        """
        :param at: Input points that define the sampler.
        :param jitter: The size of the jitter to use when stabilising the Cholesky
            decomposition of the covariance matrix.
        :return: Samples.
        """

    def reset_sampler(self) -> None:
        """
        Reset the sampler so that new samples are drawn at the next :meth:`sample` call.
        """
        self._initialized.assign(False)


TrajectoryFunction = Callable[[TensorType], TensorType]
"""
Type alias for trajectory functions.

An :const:`TrajectoryFunction` evaluates a particular sample at a set of `N` query
points (each of dimension `D`) i.e. takes input of shape `[N, D]` and returns
shape `[N, 1]`.

A key property of these trajectory functions is that the same sample draw is evaluated
for all queries. This property is known as consistency.
"""


class TrajectorySampler(ABC):
    r"""
    This class builds functions that approximate a trajectory sampled from an
    underlying :class:`ProbabilisticModel`.

    Unlike the :class:`ReparametrizationSampler`, a :class:`TrajectorySampler` provides
    consistent samples (i.e ensuring that the same sample draw is used for all evaluations
    of a particular trajectory function).
    """

    def __init__(self, model: ProbabilisticModel):
        """
        :param model: The model to sample from.
        """
        self._model = model

    def __repr__(self) -> str:
        """"""
        return f"{self.__class__.__name__}({self._model!r})"

    @abstractmethod
    def get_trajectory(self) -> TrajectoryFunction:
        """
        :return: A trajectory function representing an approximate trajectory from the
            model, taking an input of shape `[N, D]` and returning shape `[N, 1]`
        """
