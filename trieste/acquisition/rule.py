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
"""
This module contains acquisition rules, which choose the optimal point(s) to query on each step of
the Bayesian optimization process.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar, Generic, Optional, Tuple, Mapping, Union

import tensorflow as tf
from typing_extensions import Final

from ..datasets import Dataset
from ..models import ModelInterface
from ..space import SearchSpace, Box
from ..type import QueryPoints
from .function import AcquisitionFunctionBuilder, ExpectedImprovement
from . import _optimizer


S = TypeVar("S")
""" Unbound type variable. """

SP = TypeVar("SP", bound=SearchSpace, contravariant=True)
""" Contravariant type variable bound to :class:`SearchSpace`. """


class AcquisitionRule(ABC, Generic[S, SP]):
    """ The central component of the acquisition API. """

    @abstractmethod
    def acquire(
        self,
        search_space: SP,
        datasets: Mapping[str, Dataset],
        models: Mapping[str, ModelInterface],
        state: Optional[S],
    ) -> Tuple[QueryPoints, S]:
        """
        Return the optimal points within the specified ``search_space``, where optimality is defined
        by the acquisition rule.

        **Type hints:**
          - The global search space must be a :class:`~trieste.space.SearchSpace`. The exact type
            of :class:`~trieste.space.SearchSpace` depends on the specific
            :class:`AcquisitionRule`.
          - Each :class:`AcquisitionRule` must define the type of its corresponding acquisition
            state (if the rule is stateless, this type can be `None`). The ``state`` passed
            to this method, and the state returned, must both be of that type.

        :param search_space: The global search space over which the optimization problem
            is defined.
        :param datasets: The known observer query points and observations for each tag.
        :param models: The model to use for each :class:`~trieste.datasets.Dataset` in ``datasets``
            (matched by tag).
        :param state: The acquisition state from the previous step, if there was a previous step,
            else `None`.
        :return: The optimal points and the acquisition state for this step.
        """


OBJECTIVE: Final[str] = "OBJECTIVE"
"""
:var OBJECTIVE: A tag typically used by acquisition rules to denote the data sets and models
corresponding to the optimization objective.
"""


class EfficientGlobalOptimization(AcquisitionRule[None, SearchSpace]):
    """ Implements the Efficient Global Optimization, or EGO, algorithm. """

    def __init__(self, builder: Optional[AcquisitionFunctionBuilder] = None):
        """
        :param builder: The acquisition function builder to use.
            :class:`EfficientGlobalOptimization` will attempt to **maximise** the corresponding
            acquisition function. Defaults to :class:`~trieste.acquisition.ExpectedImprovement`
            with tag `OBJECTIVE`.
        """
        if builder is None:
            builder = ExpectedImprovement().using(OBJECTIVE)

        self._builder = builder

    def acquire(
        self,
        search_space: SearchSpace,
        datasets: Mapping[str, Dataset],
        models: Mapping[str, ModelInterface],
        state: None = None,
    ) -> Tuple[QueryPoints, None]:
        """
        Return the query point that optimizes the acquisition function produced by `builder` (see
        :meth:`__init__`).

        :param search_space: The global search space over which the optimization problem
            is defined.
        :param datasets: The known observer query points and observations.
        :param models: The models of the specified ``datasets``.
        :param state: Unused.
        :return: The single point to query, and `None`.
        """
        acquisition_function = self._builder.prepare_acquisition_function(datasets, models)
        point = _optimizer.optimize(search_space, acquisition_function)
        return point, None


class ThompsonSampling(AcquisitionRule[None, SearchSpace]):
    """ Implements Thompson sampling for choosing optimal points. """

    def __init__(self, num_search_space_samples: int, num_query_points: int):
        """
        :param num_search_space_samples: The number of points at which to sample the posterior.
        :param num_query_points: The number of points to acquire.
        """
        if not num_search_space_samples > 0:
            raise ValueError(f"Search space must be greater than 0, got {num_search_space_samples}")

        if not num_query_points > 0:
            raise ValueError(
                f"Number of query points must be greater than 0, got {num_query_points}"
            )

        self._num_search_space_samples = num_search_space_samples
        self._num_query_points = num_query_points

    def acquire(
        self,
        search_space: SearchSpace,
        datasets: Mapping[str, Dataset],
        models: Mapping[str, ModelInterface],
        state: None = None,
    ) -> Tuple[QueryPoints, None]:
        """
        Sample `num_search_space_samples` (see :meth:`__init__`) points from the
        ``search_space``. Of those points, return the `num_query_points` points at which
        random samples yield the **minima** of the model posterior.

        :param search_space: The global search space over which the optimization problem
            is defined.
        :param datasets: Unused.
        :param models: The model of the known data. Uses the single key `OBJECTIVE`.
        :param state: Unused.
        :return: The `num_query_points` points to query, and `None`.
        :raise ValueError: If ``models`` do not contain the key `OBJECTIVE`, or it contains any
            other key.
        """
        if models.keys() != {OBJECTIVE}:
            raise ValueError(
                f"dict of models must contain the single key {OBJECTIVE}, got keys {models.keys()}"
            )

        nqp, ns = self._num_query_points, self._num_search_space_samples
        query_points = search_space.sample(ns)  # [ns, ...]
        samples = models[OBJECTIVE].sample(query_points, nqp)  # [nqp, ns, ...]
        samples_2d = tf.reshape(samples, [nqp, ns])  # [nqp, ns]
        indices = tf.math.argmin(samples_2d, axis=1)
        unique_indices = tf.unique(indices).y
        return tf.gather(query_points, unique_indices), None


class TrustRegion(AcquisitionRule["TrustRegion.State", Box]):
    """ Implements the *trust region* acquisition algorithm. """

    @dataclass(frozen=True)
    class State:
        """
        The acquisition state for the :class:`TrustRegion` acquisition rule.

        :ivar acquisition_space: The search space.
        :ivar eps: The (maximum) vector from the current best point to each bound of the acquisition
            space.
        :ivar y_min: The minimum observed value.
        :ivar is_global: `True` if the search space was global, else `False` if it was local. May be
            a scalar boolean `tf.Tensor` instead of a `bool`.
        """

        acquisition_space: Box
        eps: tf.Tensor
        y_min: tf.Tensor
        is_global: Union[tf.Tensor, bool]

    def __init__(
        self,
        builder: Optional[AcquisitionFunctionBuilder] = None,
        beta: float = 0.7,
        kappa: float = 1e-4,
    ):
        """
        :param builder: The acquisition function builder to use. :class:`TrustRegion` will attempt
            to **maximise** the corresponding acquisition function. Defaults to
            :class:`~trieste.acquisition.ExpectedImprovement` with tag `OBJECTIVE`.
        :param beta: The inverse of the trust region contraction factor.
        :param kappa: Scales the threshold for the minimal improvement required for a step to be
            considered a success.
        """
        if builder is None:
            builder = ExpectedImprovement().using(OBJECTIVE)

        self._builder = builder
        self._beta = beta
        self._kappa = kappa

    def acquire(
        self,
        search_space: Box,
        datasets: Mapping[str, Dataset],
        models: Mapping[str, ModelInterface],
        state: Optional[State],
    ) -> Tuple[QueryPoints, State]:
        """
        Acquire one new query point according the trust region algorithm. Return the new query point
        along with the final acquisition state from this step.

        If no ``state`` is specified (it is `None`), ``search_space`` is used as
        the search space for this step.

        If a ``state`` is specified, and the new optimum improves over the previous optimum
        by some threshold (that scales linearly with ``kappa``), the previous acquisition is
        considered successful.

        If the previous acquisition was successful, ``search_space`` is used as the new
        search space. If the previous step was unsuccessful, the search space is changed to the
        trust region if it was global, and vice versa.

        If the previous acquisition was over the trust region, the size of the trust region is
        modified. If the previous acquisition was successful, the size is increased by a factor
        ``1 / beta``. Conversely, if it was unsuccessful, the size is reduced by the factor
        ``beta``.

        **Note:** The acquisition search space will never extend beyond the boundary of the
        ``search_space``. For a local search, the actual search space will be the
        intersection of the trust region and ``search_space``.

        :param search_space: The global search space for the optimization problem.
        :param datasets: The known observer query points and observations. Uses the data for key
            `OBJECTIVE` to calculate the new trust region.
        :param models: The models of the specified ``datasets``.
        :param state: The acquisition state from the previous step, if there was a previous step,
            else `None`.
        :return: A 2-tuple of the query point and the acquisition state for this step.
        :raise KeyError: If ``datasets`` does not contain the key `OBJECTIVE`.
        """
        dataset = datasets[OBJECTIVE]

        global_lower = search_space.lower
        global_upper = search_space.upper

        y_min = tf.reduce_min(dataset.observations, axis=0)

        if state is None:
            eps = 0.5 * (global_upper - global_lower) / (5.0 ** (1.0 / global_lower.shape[-1]))
            is_global = True
        else:
            tr_volume = tf.reduce_prod(
                state.acquisition_space.upper - state.acquisition_space.lower
            )
            step_is_success = y_min < state.y_min - self._kappa * tr_volume

            eps = (
                state.eps
                if state.is_global
                else state.eps / self._beta
                if step_is_success
                else state.eps * self._beta
            )

            is_global = step_is_success or not state.is_global

        if is_global:
            acquisition_space = search_space
        else:
            xmin = dataset.query_points[tf.argmin(dataset.observations)[0], :]
            acquisition_space = Box(
                tf.reduce_max([global_lower, xmin - eps], axis=0),
                tf.reduce_min([global_upper, xmin + eps], axis=0),
            )

        acquisition_function = self._builder.prepare_acquisition_function(datasets, models)
        point = _optimizer.optimize(acquisition_space, acquisition_function)
        state_ = TrustRegion.State(acquisition_space, eps, y_min, is_global)

        return point, state_
