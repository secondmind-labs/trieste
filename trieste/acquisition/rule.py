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

import copy
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar, Union

import tensorflow as tf

from .. import types
from ..data import Dataset
from ..models import ProbabilisticModel
from ..observer import OBJECTIVE
from ..space import Box, SearchSpace
from ..types import TensorType
from .function import (
    AcquisitionFunction,
    AcquisitionFunctionBuilder,
    ExpectedImprovement,
    GreedyAcquisitionFunctionBuilder,
    SingleModelAcquisitionBuilder,
    SingleModelGreedyAcquisitionBuilder,
)
from .optimizer import AcquisitionOptimizer, automatic_optimizer_selector, batchify
from .sampler import ExactThompsonSampler, RandomFourierFeatureThompsonSampler, ThompsonSampler

T_co = TypeVar("T_co", covariant=True)
""" Unbound covariant type variable. """

SP_contra = TypeVar("SP_contra", bound=SearchSpace, contravariant=True)
""" Contravariant type variable bound to :class:`~trieste.space.SearchSpace`. """


class AcquisitionRule(ABC, Generic[T_co, SP_contra]):
    """
    The central component of the acquisition API.

    An :class:`AcquisitionRule` can produce any value from the search space for this step, and the
    historic data and models. This value is typically a set of query points, either on its own as
    a `TensorType` (see e.g. :class:`EfficientGlobalOptimization`), or within some context
    (see e.g. :class:`TrustRegion`). Indeed, to use an :class:`AcquisitionRule` in the main
    :class:`~trieste.bayesian_optimizer.BayesianOptimizer` Bayesian optimization loop, the rule
    must return either a `TensorType` or `State`-ful `TensorType`.
    """

    @abstractmethod
    def acquire(
        self,
        search_space: SP_contra,
        datasets: Mapping[str, Dataset],
        models: Mapping[str, ProbabilisticModel],
    ) -> T_co:
        """
        Return a value of type `T_co`. Typically this will be a set of query points, either on its
        own as a `TensorType` (see e.g. :class:`EfficientGlobalOptimization`), or within some
        context (see e.g. :class:`TrustRegion`).

        **Type hints:**
          - The search space must be a :class:`~trieste.space.SearchSpace`. The exact type of
            :class:`~trieste.space.SearchSpace` depends on the specific :class:`AcquisitionRule`.

        :param search_space: The local acquisition search space for *this step*.
        :param datasets: The known observer query points and observations for each tag.
        :param models: The model to use for each :class:`~trieste.data.Dataset` in ``datasets``
            (matched by tag).
        :return: A value of type `T_co`.
        """

    def acquire_single(
        self,
        search_space: SP_contra,
        dataset: Dataset,
        model: ProbabilisticModel,
    ) -> T_co:
        """
        A convenience wrapper for :meth:`acquire` that uses only one model, dataset pair.

        :param search_space: The global search space over which the optimization problem
            is defined.
        :param dataset: The known observer query points and observations.
        :param model: The model to use for the dataset.
        :return: A value of type `T_co`.
        """
        if isinstance(dataset, dict) or isinstance(model, dict):
            raise ValueError(
                "AcquisitionRule.acquire_single method does not support multiple datasets "
                "or models: use acquire instead"
            )
        return self.acquire(search_space, {OBJECTIVE: dataset}, {OBJECTIVE: model})


class EfficientGlobalOptimization(AcquisitionRule[TensorType, SP_contra]):
    """Implements the Efficient Global Optimization, or EGO, algorithm."""

    def __init__(
        self,
        builder: Optional[
            AcquisitionFunctionBuilder
            | GreedyAcquisitionFunctionBuilder
            | SingleModelAcquisitionBuilder
            | SingleModelGreedyAcquisitionBuilder
        ] = None,
        optimizer: AcquisitionOptimizer[SP_contra] | None = None,
        num_query_points: int = 1,
    ):
        """
        :param builder: The acquisition function builder to use. Defaults to
            :class:`~trieste.acquisition.ExpectedImprovement`.
        :param optimizer: The optimizer with which to optimize the acquisition function built by
            ``builder``. This should *maximize* the acquisition function, and must be compatible
            with the global search space. Defaults to
            :func:`~trieste.acquisition.optimizer.automatic_optimizer_selector`.
        :param num_query_points: The number of points to acquire.
        """

        if num_query_points <= 0:
            raise ValueError(
                f"Number of query points must be greater than 0, got {num_query_points}"
            )

        if builder is None:
            if num_query_points == 1:
                builder = ExpectedImprovement()
            else:
                raise ValueError(
                    """Need to specify a batch acquisition function when number of query points
                    is greater than 1"""
                )

        if optimizer is None:
            optimizer = automatic_optimizer_selector

        if isinstance(
            builder, (SingleModelAcquisitionBuilder, SingleModelGreedyAcquisitionBuilder)
        ):
            builder = builder.using(OBJECTIVE)

        if isinstance(builder, AcquisitionFunctionBuilder):
            # Joint batch acquisitions require batch optimizers
            optimizer = batchify(optimizer, num_query_points)

        self._builder: Union[AcquisitionFunctionBuilder, GreedyAcquisitionFunctionBuilder] = builder
        self._optimizer = optimizer
        self._num_query_points = num_query_points
        self._acquisition_function: Optional[AcquisitionFunction] = None

    def __repr__(self) -> str:
        """"""
        return f"""EfficientGlobalOptimization(
        {self._builder!r},
        {self._optimizer!r},
        {self._num_query_points!r})"""

    def acquire(
        self,
        search_space: SP_contra,
        datasets: Mapping[str, Dataset],
        models: Mapping[str, ProbabilisticModel],
    ) -> TensorType:
        """
        Return the query point(s) that optimizes the acquisition function produced by ``builder``
        (see :meth:`__init__`).

        :param search_space: The local acquisition search space for *this step*.
        :param datasets: The known observer query points and observations.
        :param models: The models of the specified ``datasets``.
        :return: The single (or batch of) points to query.
        """
        if self._acquisition_function is None:
            self._acquisition_function = self._builder.prepare_acquisition_function(
                datasets, models
            )
        else:
            self._acquisition_function = self._builder.update_acquisition_function(
                self._acquisition_function, datasets, models
            )

        points = self._optimizer(search_space, self._acquisition_function)

        if isinstance(self._builder, GreedyAcquisitionFunctionBuilder):
            for _ in range(
                self._num_query_points - 1
            ):  # greedily allocate remaining batch elements
                self._acquisition_function = self._builder.update_acquisition_function(
                    self._acquisition_function, datasets, models, pending_points=points
                )
                chosen_point = self._optimizer(search_space, self._acquisition_function)
                points = tf.concat([points, chosen_point], axis=0)

        return points


class DiscreteThompsonSampling(AcquisitionRule[TensorType, SearchSpace]):
    r"""
    Implements Thompson sampling for choosing optimal points.

    This rule returns the minimizers of functions sampled from our model and evaluated across
    a discretization of the search space (containing `N` candidate points).

    The model is sampled either exactly (with an :math:`O(N^3)` complexity), or sampled
    approximately through a random Fourier `M` feature decompisition
    (with an :math:`O(\min(n^3,M^3))` complexity for a model trained on `n` points).

    """

    def __init__(
        self,
        num_search_space_samples: int,
        num_query_points: int,
        num_fourier_features: Optional[int] = None,
    ):
        """
        :param num_search_space_samples: The number of points at which to sample the posterior.
        :param num_query_points: The number of points to acquire.
        :num_fourier_features: The number of features used to approximate the kernel. We
            recommend first trying 1000 features, as this typically perfoms well for a wide
            range of kernels. If None, then we perfom exact Thompson sampling.
        """
        if not num_search_space_samples > 0:
            raise ValueError(f"Search space must be greater than 0, got {num_search_space_samples}")

        if not num_query_points > 0:
            raise ValueError(
                f"Number of query points must be greater than 0, got {num_query_points}"
            )

        if num_fourier_features is not None and num_fourier_features <= 0:
            raise ValueError(
                f"Number of fourier features must be greater than 0, got {num_query_points}"
            )

        self._num_search_space_samples = num_search_space_samples
        self._num_query_points = num_query_points
        self._num_fourier_features = num_fourier_features

    def __repr__(self) -> str:
        """"""
        return f"""DiscreteThompsonSampling(
        {self._num_search_space_samples!r},
        {self._num_query_points!r},
        {self._num_fourier_features!r})"""

    def acquire(
        self,
        search_space: SearchSpace,
        datasets: Mapping[str, Dataset],
        models: Mapping[str, ProbabilisticModel],
    ) -> TensorType:
        """
        Sample `num_search_space_samples` (see :meth:`__init__`) points from the
        ``search_space``. Of those points, return the `num_query_points` points at which
        random samples yield the **minima** of the model posterior.

        :param search_space: The local acquisition search space for *this step*.
        :param datasets: Unused.
        :param models: The model of the known data. Uses the single key `OBJECTIVE`.
        :return: The ``num_query_points`` points to query.
        :raise ValueError: If ``models`` do not contain the key `OBJECTIVE`, or it contains any
            other key.
        """
        if models.keys() != {OBJECTIVE}:
            raise ValueError(
                f"dict of models must contain the single key {OBJECTIVE}, got keys {models.keys()}"
            )

        if datasets.keys() != {OBJECTIVE}:
            raise ValueError(
                f"""
                dict of datasets must contain the single key {OBJECTIVE},
                got keys {datasets.keys()}
                """
            )

        if self._num_fourier_features is None:  # Perform exact Thompson sampling
            thompson_sampler: ThompsonSampler = ExactThompsonSampler(
                self._num_query_points, models[OBJECTIVE]
            )
        else:  # Perform approximate Thompson sampling
            thompson_sampler = RandomFourierFeatureThompsonSampler(
                self._num_query_points,
                models[OBJECTIVE],
                datasets[OBJECTIVE],
                num_features=self._num_fourier_features,
            )

        query_points = search_space.sample(self._num_search_space_samples)
        thompson_samples = thompson_sampler.sample(query_points)

        return thompson_samples


class TrustRegion(AcquisitionRule[types.State[Optional["TrustRegion.State"], TensorType], Box]):
    """Implements the *trust region* acquisition algorithm."""

    @dataclass(frozen=True)
    class State:
        """The acquisition state for the :class:`TrustRegion` acquisition rule."""

        acquisition_space: Box
        """ The search space. """

        eps: TensorType
        """
        The (maximum) vector from the current best point to each bound of the acquisition space.
        """

        y_min: TensorType
        """ The minimum observed value. """

        is_global: bool | TensorType
        """
        `True` if the search space was global, else `False` if it was local. May be a scalar boolean
        `TensorType` instead of a `bool`.
        """

        def __deepcopy__(self, memo: dict[int, object]) -> TrustRegion.State:
            box_copy = copy.deepcopy(self.acquisition_space, memo)
            return TrustRegion.State(box_copy, self.eps, self.y_min, self.is_global)

    def __init__(
        self,
        rule: AcquisitionRule[TensorType, Box] | None = None,
        beta: float = 0.7,
        kappa: float = 1e-4,
    ):
        """
        :param rule: The acquisition rule that defines how to search for a new query point in a
            given search space. Defaults to :class:`EfficientGlobalOptimization` with default
            arguments.
        :param beta: The inverse of the trust region contraction factor.
        :param kappa: Scales the threshold for the minimal improvement required for a step to be
            considered a success.
        """
        if rule is None:
            rule = EfficientGlobalOptimization()

        self._rule = rule
        self._beta = beta
        self._kappa = kappa

    def __repr__(self) -> str:
        """"""
        return f"TrustRegion({self._rule!r}, {self._beta!r}, {self._kappa!r})"

    def acquire(
        self,
        search_space: Box,
        datasets: Mapping[str, Dataset],
        models: Mapping[str, ProbabilisticModel],
    ) -> types.State[State | None, TensorType]:
        """
        Construct a local search space from ``search_space`` according the trust region algorithm,
        and use that with the ``rule`` specified at :meth:`~TrustRegion.__init__` to find new
        query points. Return a function that constructs these points given a previous trust region
        state.

        If no ``state`` is specified (it is `None`), ``search_space`` is used as the search space
        for this step.

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

        :param search_space: The local acquisition search space for *this step*.
        :param datasets: The known observer query points and observations. Uses the data for key
            `OBJECTIVE` to calculate the new trust region.
        :param models: The models of the specified ``datasets``.
        :return: A function that constructs the next acquisition state and the recommended query
            points from the previous acquisition state.
        :raise KeyError: If ``datasets`` does not contain the key `OBJECTIVE`.
        """
        dataset = datasets[OBJECTIVE]

        global_lower = search_space.lower
        global_upper = search_space.upper

        y_min = tf.reduce_min(dataset.observations, axis=0)

        def go(state: TrustRegion.State | None) -> tuple[TrustRegion.State | None, TensorType]:

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

            points = self._rule.acquire(acquisition_space, datasets, models)
            state_ = TrustRegion.State(acquisition_space, eps, y_min, is_global)

            return state_, points

        return go
