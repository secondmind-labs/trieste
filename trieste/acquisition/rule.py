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
from typing import Generic, Optional, TypeVar, Union, cast, overload

import tensorflow as tf

from .. import types
from ..data import Dataset
from ..logging import get_step_number, get_tensorboard_writer
from ..models import ProbabilisticModel
from ..models.interfaces import HasReparamSampler, ProbabilisticModelType
from ..observer import OBJECTIVE
from ..space import Box, SearchSpace
from ..types import State, TensorType
from .function import BatchMonteCarloExpectedImprovement, ExpectedImprovement
from .interface import (
    AcquisitionFunction,
    AcquisitionFunctionBuilder,
    GreedyAcquisitionFunctionBuilder,
    SingleModelAcquisitionBuilder,
    SingleModelGreedyAcquisitionBuilder,
    SingleModelVectorizedAcquisitionBuilder,
    VectorizedAcquisitionFunctionBuilder,
)
from .optimizer import (
    AcquisitionOptimizer,
    automatic_optimizer_selector,
    batchify_joint,
    batchify_vectorize,
)
from .sampler import ExactThompsonSampler, ThompsonSampler

ResultType = TypeVar("ResultType", covariant=True)
""" Unbound covariant type variable. """

SearchSpaceType = TypeVar("SearchSpaceType", bound=SearchSpace, contravariant=True)
""" Contravariant type variable bound to :class:`~trieste.space.SearchSpace`. """


class AcquisitionRule(ABC, Generic[ResultType, SearchSpaceType, ProbabilisticModelType]):
    """
    The central component of the acquisition API.

    An :class:`AcquisitionRule` can produce any value from the search space for this step, and the
    historic data and models. This value is typically a set of query points, either on its own as
    a `TensorType` (see e.g. :class:`EfficientGlobalOptimization`), or within some context
    (see e.g. :class:`TrustRegion`). Indeed, to use an :class:`AcquisitionRule` in the main
    :class:`~trieste.bayesian_optimizer.BayesianOptimizer` Bayesian optimization loop, the rule
    must return either a `TensorType` or `State`-ful `TensorType`.

    Note that an :class:`AcquisitionRule` might only support models with specific features (for
    example, if it uses an acquisition function that relies on those features). The type of
    models supported by a rule is indicated by the generic type variable
    class:`ProbabilisticModelType`.
    """

    @abstractmethod
    def acquire(
        self,
        search_space: SearchSpaceType,
        models: Mapping[str, ProbabilisticModelType],
        datasets: Optional[Mapping[str, Dataset]] = None,
    ) -> ResultType:
        """
        Return a value of type `T_co`. Typically this will be a set of query points, either on its
        own as a `TensorType` (see e.g. :class:`EfficientGlobalOptimization`), or within some
        context (see e.g. :class:`TrustRegion`). We assume that this requires at least models, but
        it may sometimes also need data.

        **Type hints:**
          - The search space must be a :class:`~trieste.space.SearchSpace`. The exact type of
            :class:`~trieste.space.SearchSpace` depends on the specific :class:`AcquisitionRule`.

        :param search_space: The local acquisition search space for *this step*.
        :param models: The model for each tag.
        :param datasets: The known observer query points and observations for each tag (optional).
        :return: A value of type `T_co`.
        """

    def acquire_single(
        self,
        search_space: SearchSpaceType,
        model: ProbabilisticModelType,
        dataset: Optional[Dataset] = None,
    ) -> ResultType:
        """
        A convenience wrapper for :meth:`acquire` that uses only one model, dataset pair.

        :param search_space: The global search space over which the optimization problem
            is defined.
        :param model: The model to use.
        :param dataset: The known observer query points and observations (optional).
        :return: A value of type `T_co`.
        """
        if isinstance(dataset, dict) or isinstance(model, dict):
            raise ValueError(
                "AcquisitionRule.acquire_single method does not support multiple datasets "
                "or models: use acquire instead"
            )
        return self.acquire(
            search_space,
            {OBJECTIVE: model},
            datasets=None if dataset is None else {OBJECTIVE: dataset},
        )


class EfficientGlobalOptimization(
    AcquisitionRule[TensorType, SearchSpaceType, ProbabilisticModelType]
):
    """Implements the Efficient Global Optimization, or EGO, algorithm."""

    @overload
    def __init__(
        self: "EfficientGlobalOptimization[SearchSpaceType, ProbabilisticModel]",
        builder: None = None,
        optimizer: AcquisitionOptimizer[SearchSpaceType] | None = None,
        num_query_points: int = 1,
    ):
        ...

    @overload
    def __init__(
        self: "EfficientGlobalOptimization[SearchSpaceType, ProbabilisticModelType]",
        builder: (
            AcquisitionFunctionBuilder[ProbabilisticModelType]
            | GreedyAcquisitionFunctionBuilder[ProbabilisticModelType]
            | SingleModelAcquisitionBuilder[ProbabilisticModelType]
            | SingleModelGreedyAcquisitionBuilder[ProbabilisticModelType]
        ),
        optimizer: AcquisitionOptimizer[SearchSpaceType] | None = None,
        num_query_points: int = 1,
    ):
        ...

    def __init__(
        self,
        builder: Optional[
            AcquisitionFunctionBuilder[ProbabilisticModelType]
            | GreedyAcquisitionFunctionBuilder[ProbabilisticModelType]
            | VectorizedAcquisitionFunctionBuilder[ProbabilisticModelType]
            | SingleModelAcquisitionBuilder[ProbabilisticModelType]
            | SingleModelGreedyAcquisitionBuilder[ProbabilisticModelType]
            | SingleModelVectorizedAcquisitionBuilder[ProbabilisticModelType]
        ] = None,
        optimizer: AcquisitionOptimizer[SearchSpaceType] | None = None,
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
            builder,
            (
                SingleModelAcquisitionBuilder,
                SingleModelGreedyAcquisitionBuilder,
                SingleModelVectorizedAcquisitionBuilder,
            ),
        ):
            builder = builder.using(OBJECTIVE)

        if num_query_points > 1:  # need to build batches of points
            if isinstance(builder, VectorizedAcquisitionFunctionBuilder):
                # optimize batch elements independently
                optimizer = batchify_vectorize(optimizer, num_query_points)
            elif isinstance(builder, AcquisitionFunctionBuilder):
                # optimize batch elements jointly
                optimizer = batchify_joint(optimizer, num_query_points)
            elif isinstance(builder, GreedyAcquisitionFunctionBuilder):
                # optimize batch elements sequentially using the logic in acquire.
                pass

        self._builder: Union[
            AcquisitionFunctionBuilder[ProbabilisticModelType],
            GreedyAcquisitionFunctionBuilder[ProbabilisticModelType],
            VectorizedAcquisitionFunctionBuilder[ProbabilisticModelType],
        ] = builder
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
        search_space: SearchSpaceType,
        models: Mapping[str, ProbabilisticModelType],
        datasets: Optional[Mapping[str, Dataset]] = None,
    ) -> TensorType:
        """
        Return the query point(s) that optimizes the acquisition function produced by ``builder``
        (see :meth:`__init__`).

        :param search_space: The local acquisition search space for *this step*.
        :param models: The model for each tag.
        :param datasets: The known observer query points and observations. Whether this is required
            depends on the acquisition function used.
        :return: The single (or batch of) points to query.
        """
        if self._acquisition_function is None:
            self._acquisition_function = self._builder.prepare_acquisition_function(
                models,
                datasets=datasets,
            )
        else:
            self._acquisition_function = self._builder.update_acquisition_function(
                self._acquisition_function,
                models,
                datasets=datasets,
            )

        points = self._optimizer(search_space, self._acquisition_function)

        summary_writer = get_tensorboard_writer()
        step_number = get_step_number()
        if summary_writer:
            with summary_writer.as_default(step=step_number):
                batched_points = tf.expand_dims(points, axis=0)
                value = self._acquisition_function(batched_points)[0][0]
                tf.summary.scalar("EGO.acquisition_function.maximum_found", value)

        if isinstance(self._builder, GreedyAcquisitionFunctionBuilder):
            for i in range(
                self._num_query_points - 1
            ):  # greedily allocate remaining batch elements
                self._acquisition_function = self._builder.update_acquisition_function(
                    self._acquisition_function,
                    models,
                    datasets=datasets,
                    pending_points=points,
                    new_optimization_step=False,
                )
                chosen_point = self._optimizer(search_space, self._acquisition_function)
                points = tf.concat([points, chosen_point], axis=0)

                if summary_writer:
                    with summary_writer.as_default(step=step_number):
                        batched_points = tf.expand_dims(chosen_point, axis=0)
                        value = self._acquisition_function(batched_points)[0][0]
                        tf.summary.scalar(f"EGO.acquisition_function.maximum_found.{i+1}", value)

        return points


@dataclass(frozen=True)
class AsynchronousRuleState:
    """Stores pending points for asynchronous rules.
    These are points which were requested but are not observed yet.
    """

    pending_points: Optional[TensorType] = None

    def __post_init__(self) -> None:
        if self.pending_points is None:
            # that's fine, no validation needed
            return

        tf.debugging.assert_shapes(
            [(self.pending_points, ["N", "D"])],
            message=f"""Pending points are expected to be a 2D tensor,
                        instead received tensor of shape {tf.shape(self.pending_points)}""",
        )

    @property
    def has_pending_points(self) -> bool:
        """Returns `True` if there is at least one pending point, and `False` otherwise."""
        return (self.pending_points is not None) and tf.size(self.pending_points) > 0

    def remove_points(self, points_to_remove: TensorType) -> AsynchronousRuleState:
        """Removes all rows from current `pending_points` that are present in `points_to_remove`.
        If a point to remove occurs multiple times in the list of pending points,
        only first occurrence of it will be removed.

        :param points_to_remove: Points to remove.
        :return: New instance of `AsynchronousRuleState` with updated pending points.
        """

        @tf.function
        def _remove_point(pending_points: TensorType, point_to_remove: TensorType) -> TensorType:
            # find all points equal to the one we need to remove
            are_points_equal = tf.reduce_all(tf.equal(pending_points, point_to_remove), axis=1)
            if not tf.reduce_any(are_points_equal):
                # point to remove isn't there, nothing to do
                return pending_points

            # this line converts all bool values to 0 and 1
            # then finds first 1 and returns its index as 1x1 tensor
            _, first_index_tensor = tf.math.top_k(tf.cast(are_points_equal, tf.int8), k=1)
            # to use it as index for slicing, we need to convert 1x1 tensor to a TF scalar
            first_index = tf.reshape(first_index_tensor, [])
            return tf.concat(
                [pending_points[:first_index, :], pending_points[first_index + 1 :, :]], axis=0
            )

        if not self.has_pending_points:
            # nothing to do if there are no pending points
            return self

        tf.debugging.assert_shapes(
            [(self.pending_points, [None, "D"]), (points_to_remove, [None, "D"])],
            message=f"""Point to remove shall be 1xD where D is the last dimension of pending points.
                        Got {tf.shape(self.pending_points)} for pending points
                        and {tf.shape(points_to_remove)} for other points.""",
        )

        new_pending_points = tf.foldl(
            _remove_point, points_to_remove, initializer=self.pending_points
        )
        return AsynchronousRuleState(new_pending_points)

    def add_pending_points(self, new_points: TensorType) -> AsynchronousRuleState:
        """Adds `new_points` to the already known pending points.

        :param new_points: Points to add.
        :return: New instance of `AsynchronousRuleState` with updated pending points.
        """
        if not self.has_pending_points:
            return AsynchronousRuleState(new_points)

        tf.debugging.assert_shapes(
            [(self.pending_points, [None, "D"]), (new_points, [None, "D"])],
            message=f"""New points shall be 2D and have same last dimension as pending points.
                        Got {tf.shape(self.pending_points)} for pending points
                        and {tf.shape(new_points)} for new points.""",
        )

        new_pending_points = tf.concat([self.pending_points, new_points], axis=0)
        return AsynchronousRuleState(new_pending_points)


class AsynchronousOptimization(
    AcquisitionRule[
        State[Optional["AsynchronousRuleState"], TensorType],
        SearchSpaceType,
        ProbabilisticModelType,
    ]
):
    """AsynchronousOptimization rule is designed for asynchronous BO scenarios.
    By asynchronous BO we understand a use case when multiple objective function
    can be launched in parallel and are expected to arrive at different times.
    Instead of waiting for the rest of observations to return, we want to immediately
    use acquisition function to launch a new observation and avoid wasting computational resources.
    See :cite:`Alvi:2019` or :cite:`kandasamy18a` for more details.

    To make the best decision about next point to observe, acquisition function
    needs to be aware of currently running observations.
    We call such points "pending", and consider them a part of acquisition state.
    We use :class:`AsynchronousRuleState` to store these points.

    `AsynchronousOptimization` works with non-greedy batch acquisition functions.
    For example, it would work with
    :class:`~trieste.acquisition.BatchMonteCarloExpectedImprovement`,
    but cannot be used with :class:`~trieste.acquisition.ExpectedImprovement`.
    If there are P pending points and the batch of size B is requested,
    the acquisition function is used with batch size P+B.
    During optimization first P points are fixed to pending,
    and thus we optimize and return the last B points only.
    """

    @overload
    def __init__(
        self: "AsynchronousOptimization[SearchSpaceType, HasReparamSampler]",
        builder: None = None,
        optimizer: AcquisitionOptimizer[SearchSpaceType] | None = None,
        num_query_points: int = 1,
    ):
        ...

    @overload
    def __init__(
        self: "AsynchronousOptimization[SearchSpaceType, ProbabilisticModelType]",
        builder: (
            AcquisitionFunctionBuilder[ProbabilisticModelType]
            | SingleModelAcquisitionBuilder[ProbabilisticModelType]
        ),
        optimizer: AcquisitionOptimizer[SearchSpaceType] | None = None,
        num_query_points: int = 1,
    ):
        ...

    def __init__(
        self,
        builder: Optional[
            AcquisitionFunctionBuilder[ProbabilisticModelType]
            | SingleModelAcquisitionBuilder[ProbabilisticModelType]
        ] = None,
        optimizer: AcquisitionOptimizer[SearchSpaceType] | None = None,
        num_query_points: int = 1,
    ):
        """
        :param builder: Batch acquisition function builder. Defaults to
            :class:`~trieste.acquisition.BatchMonteCarloExpectedImprovement` with 10 000 samples.
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
            builder = cast(
                SingleModelAcquisitionBuilder[ProbabilisticModelType],
                BatchMonteCarloExpectedImprovement(10_000),
            )

        if optimizer is None:
            optimizer = automatic_optimizer_selector

        if isinstance(builder, SingleModelAcquisitionBuilder):
            builder = builder.using(OBJECTIVE)

        # even though we are only using batch acquisition functions
        # there is no need to batchify_joint the optimizer if our batch size is 1
        if num_query_points > 1:
            optimizer = batchify_joint(optimizer, num_query_points)

        self._builder: AcquisitionFunctionBuilder[ProbabilisticModelType] = builder
        self._optimizer = optimizer
        self._acquisition_function: Optional[AcquisitionFunction] = None

    def __repr__(self) -> str:
        """"""
        return f"""AsynchronousOptimization(
        {self._builder!r},
        {self._optimizer!r})"""

    def acquire(
        self,
        search_space: SearchSpaceType,
        models: Mapping[str, ProbabilisticModelType],
        datasets: Optional[Mapping[str, Dataset]] = None,
    ) -> types.State[AsynchronousRuleState | None, TensorType]:
        """
        Constructs a function that, given ``AsynchronousRuleState``,
        returns a new state object and points to evaluate.
        The state object contains currently known pending points,
        that is points that were requested for evaluation,
        but observation for which was not received yet.
        To keep them up to date, pending points are compared against the given dataset,
        and whatever points are in the dataset are deleted.

        Let's suppose we have P pending points. To optimize the acquisition function
        we call it with batches of size P+1, where first P points are fixed to pending points.
        Optimization therefore happens over the last point only, which is returned.

        :param search_space: The local acquisition search space for *this step*.
        :param models: The model of the known data. Uses the single key `OBJECTIVE`.
        :param datasets: The known observer query points and observations.
        :return: A function that constructs the next acquisition state and the recommended query
            points from the previous acquisition state.
        """
        if models.keys() != {OBJECTIVE}:
            raise ValueError(
                f"dict of models must contain the single key {OBJECTIVE}, got keys {models.keys()}"
            )
        if datasets is None or datasets.keys() != {OBJECTIVE}:
            raise ValueError(
                f"""datasets must be provided and contain the single key {OBJECTIVE}"""
            )

        if self._acquisition_function is None:
            self._acquisition_function = self._builder.prepare_acquisition_function(
                models,
                datasets=datasets,
            )
        else:
            self._acquisition_function = self._builder.update_acquisition_function(
                self._acquisition_function,
                models,
                datasets=datasets,
            )

        def state_func(
            state: AsynchronousRuleState | None,
        ) -> tuple[AsynchronousRuleState | None, TensorType]:
            tf.debugging.Assert(self._acquisition_function is not None, [])

            if state is None:
                state = AsynchronousRuleState(None)

            assert datasets is not None
            state = state.remove_points(datasets[OBJECTIVE].query_points)

            if state.has_pending_points:
                pending_points: TensorType = state.pending_points

                def function_with_pending_points(x: TensorType) -> TensorType:
                    # stuff below is quite tricky, and thus deserves an elaborate comment
                    # we receive unknown number N of batches to evaluate
                    # and need to collect batch of B new points
                    # so the shape of `x` is [N, B, D]
                    # we want to add P pending points to each batch
                    # so that acquisition actually receives N batches of shape [P+B, D] each
                    # therefore here we prepend each batch with all pending points
                    # resulting a shape [N, P+B, D]
                    # we do that by repeating pending points N times and concatenating with x

                    # pending points are 2D, we need 3D and repeat along first axis
                    expanded = tf.expand_dims(pending_points, axis=0)
                    pending_points_repeated = tf.repeat(expanded, [tf.shape(x)[0]], axis=0)
                    all_points = tf.concat([pending_points_repeated, x], axis=1)
                    return cast(AcquisitionFunction, self._acquisition_function)(all_points)

                acquisition_function = cast(AcquisitionFunction, function_with_pending_points)
            else:
                acquisition_function = cast(AcquisitionFunction, self._acquisition_function)

            new_points = self._optimizer(search_space, acquisition_function)
            state = state.add_pending_points(new_points)

            return state, new_points

        return state_func


class AsynchronousGreedy(
    AcquisitionRule[
        State[Optional["AsynchronousRuleState"], TensorType],
        SearchSpaceType,
        ProbabilisticModelType,
    ]
):
    """AsynchronousGreedy rule, as name suggests,
    is designed for asynchronous BO scenarios. To see what we understand by
    asynchronous BO, see documentation for :class:`~trieste.acquisition.AsynchronousOptimization`.

    AsynchronousGreedy rule works with greedy batch acquisition functions
    and performs B steps of a greedy batch collection process,
    where B is the requested batch size.
    """

    def __init__(
        self,
        builder: GreedyAcquisitionFunctionBuilder[ProbabilisticModelType]
        | SingleModelGreedyAcquisitionBuilder[ProbabilisticModelType],
        optimizer: AcquisitionOptimizer[SearchSpaceType] | None = None,
        num_query_points: int = 1,
    ):
        """
        :param builder: Acquisition function builder. Only greedy batch approaches are supported,
            because they can be told what points are pending.
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
            raise ValueError("Please specify an acquisition builder")

        if not isinstance(
            builder, (GreedyAcquisitionFunctionBuilder, SingleModelGreedyAcquisitionBuilder)
        ):
            raise NotImplementedError(
                f"""Only greedy acquisition strategies are supported,
                    got {type(builder)}"""
            )

        if optimizer is None:
            optimizer = automatic_optimizer_selector

        if isinstance(builder, SingleModelGreedyAcquisitionBuilder):
            builder = builder.using(OBJECTIVE)

        self._builder: GreedyAcquisitionFunctionBuilder[ProbabilisticModelType] = builder
        self._optimizer = optimizer
        self._acquisition_function: Optional[AcquisitionFunction] = None
        self._num_query_points = num_query_points

    def __repr__(self) -> str:
        """"""
        return f"""AsynchronousGreedy(
        {self._builder!r},
        {self._optimizer!r},
        {self._num_query_points!r})"""

    def acquire(
        self,
        search_space: SearchSpaceType,
        models: Mapping[str, ProbabilisticModelType],
        datasets: Optional[Mapping[str, Dataset]] = None,
    ) -> types.State[AsynchronousRuleState | None, TensorType]:
        """
        Constructs a function that, given ``AsynchronousRuleState``,
        returns a new state object and points to evaluate.
        The state object contains currently known pending points,
        that is points that were requested for evaluation,
        but observation for which was not received yet.
        To keep them up to date, pending points are compared against the given dataset,
        and whatever points are in the dataset are deleted.
        Then the current batch is generated by calling the acquisition function,
        and all points in the batch are added to the known pending points.

        :param search_space: The local acquisition search space for *this step*.
        :param models: The model of the known data. Uses the single key `OBJECTIVE`.
        :param datasets: The known observer query points and observations.
        :return: A function that constructs the next acquisition state and the recommended query
            points from the previous acquisition state.
        """
        if models.keys() != {OBJECTIVE}:
            raise ValueError(
                f"dict of models must contain the single key {OBJECTIVE}, got keys {models.keys()}"
            )
        if datasets is None or datasets.keys() != {OBJECTIVE}:
            raise ValueError(
                f"""datasets must be provided and contain the single key {OBJECTIVE}"""
            )

        def state_func(
            state: AsynchronousRuleState | None,
        ) -> tuple[AsynchronousRuleState | None, TensorType]:
            if state is None:
                state = AsynchronousRuleState(None)

            assert datasets is not None
            state = state.remove_points(datasets[OBJECTIVE].query_points)

            if self._acquisition_function is None:
                self._acquisition_function = self._builder.prepare_acquisition_function(
                    models,
                    datasets=datasets,
                    pending_points=state.pending_points,
                )
            else:
                self._acquisition_function = self._builder.update_acquisition_function(
                    self._acquisition_function,
                    models,
                    datasets=datasets,
                    pending_points=state.pending_points,
                )

            new_points_batch = self._optimizer(search_space, self._acquisition_function)
            state = state.add_pending_points(new_points_batch)

            summary_writer = get_tensorboard_writer()
            step_number = get_step_number()

            for i in range(self._num_query_points - 1):
                # greedily allocate additional batch elements
                self._acquisition_function = self._builder.update_acquisition_function(
                    self._acquisition_function,
                    models,
                    datasets=datasets,
                    pending_points=state.pending_points,
                    new_optimization_step=False,
                )
                new_point = self._optimizer(search_space, self._acquisition_function)
                if summary_writer:
                    with summary_writer.as_default(step=step_number):
                        batched_point = tf.expand_dims(new_point, axis=0)
                        value = self._acquisition_function(batched_point)[0][0]
                        tf.summary.scalar(
                            f"AsyncGreedy.acquisition_function.maximum_found.{i}", value
                        )
                state = state.add_pending_points(new_point)
                new_points_batch = tf.concat([new_points_batch, new_point], axis=0)

            return state, new_points_batch

        return state_func


class RandomSampling(AcquisitionRule[TensorType, SearchSpace, ProbabilisticModel]):
    """
    This class performs random search for choosing optimal points. It uses ``sample`` method
    from :class:`~trieste.space.SearchSpace` to take random samples from the search space that
    are used as optimal points. Hence, it does not use any acquisition function. This
    acquisition rule can be useful as a baseline for other acquisition functions of interest.
    """

    def __init__(self, num_query_points: int = 1):
        """
        :param num_query_points: The number of points to acquire. By default set to 1 point.
        :raise ValueError: If ``num_query_points`` is less or equal to 0.
        """
        if num_query_points <= 0:
            raise ValueError(
                f"Number of query points must be greater than 0, got {num_query_points}"
            )
        self._num_query_points = num_query_points

    def __repr__(self) -> str:
        """"""
        return f"""RandomSampling({self._num_query_points!r})"""

    def acquire(
        self,
        search_space: SearchSpace,
        models: Mapping[str, ProbabilisticModel],
        datasets: Optional[Mapping[str, Dataset]] = None,
    ) -> TensorType:
        """
        Sample ``num_query_points`` (see :meth:`__init__`) points from the
        ``search_space``.

        :param search_space: The acquisition search space.
        :param models: Unused.
        :param datasets: Unused.
        :return: The ``num_query_points`` points to query.
        """
        samples = search_space.sample(self._num_query_points)

        return samples


class DiscreteThompsonSampling(AcquisitionRule[TensorType, SearchSpace, ProbabilisticModelType]):
    r"""
    Implements Thompson sampling for choosing optimal points.

    This rule returns the minimizers of functions sampled from our model and evaluated across
    a discretization of the search space (containing `N` candidate points).

    The model is sampled either exactly (with an :math:`O(N^3)` complexity), or sampled
    approximately through a random Fourier `M` feature decompisition
    (with an :math:`O(\min(n^3,M^3))` complexity for a model trained on `n` points). The number
    `M` of Fourier features is specified when building the model.

    """

    @overload
    def __init__(
        self: "DiscreteThompsonSampling[ProbabilisticModel]",
        num_search_space_samples: int,
        num_query_points: int,
        thompson_sampler: None = None,
    ):
        ...

    @overload
    def __init__(
        self: "DiscreteThompsonSampling[ProbabilisticModelType]",
        num_search_space_samples: int,
        num_query_points: int,
        thompson_sampler: Optional[ThompsonSampler[ProbabilisticModelType]] = None,
    ):
        ...

    def __init__(
        self,
        num_search_space_samples: int,
        num_query_points: int,
        thompson_sampler: Optional[ThompsonSampler[ProbabilisticModelType]] = None,
    ):
        """
        :param num_search_space_samples: The number of points at which to sample the posterior.
        :param num_query_points: The number of points to acquire.
        :thompson_sampler: Sampler to sample maximisers from the underlying model.
        """
        if not num_search_space_samples > 0:
            raise ValueError(f"Search space must be greater than 0, got {num_search_space_samples}")

        if not num_query_points > 0:
            raise ValueError(
                f"Number of query points must be greater than 0, got {num_query_points}"
            )

        if thompson_sampler is not None:
            if thompson_sampler.sample_min_value:
                raise ValueError(
                    """
                    Thompson sampling requires a thompson_sampler that samples minimizers,
                    not just minimum values. However the passed sampler has sample_min_value=True.
                    """
                )
        else:
            thompson_sampler = ExactThompsonSampler(sample_min_value=False)

        self._thompson_sampler = thompson_sampler
        self._num_search_space_samples = num_search_space_samples
        self._num_query_points = num_query_points

    def __repr__(self) -> str:
        """"""
        return f"""DiscreteThompsonSampling(
        {self._num_search_space_samples!r},
        {self._num_query_points!r},
        {self._thompson_sampler!r})"""

    def acquire(
        self,
        search_space: SearchSpace,
        models: Mapping[str, ProbabilisticModelType],
        datasets: Optional[Mapping[str, Dataset]] = None,
    ) -> TensorType:
        """
        Sample `num_search_space_samples` (see :meth:`__init__`) points from the
        ``search_space``. Of those points, return the `num_query_points` points at which
        random samples yield the **minima** of the model posterior.

        :param search_space: The local acquisition search space for *this step*.
        :param models: The model of the known data. Uses the single key `OBJECTIVE`.
        :param datasets: The known observer query points and observations.
        :return: The ``num_query_points`` points to query.
        :raise ValueError: If ``models`` do not contain the key `OBJECTIVE`, or it contains any
            other key.
        """
        if models.keys() != {OBJECTIVE}:
            raise ValueError(
                f"dict of models must contain the single key {OBJECTIVE}, got keys {models.keys()}"
            )

        if datasets is None or datasets.keys() != {OBJECTIVE}:
            raise ValueError(
                f"""datasets must be provided and contain the single key {OBJECTIVE}"""
            )

        query_points = search_space.sample(self._num_search_space_samples)
        thompson_samples = self._thompson_sampler.sample(
            models[OBJECTIVE], self._num_query_points, query_points
        )

        return thompson_samples


class TrustRegion(
    AcquisitionRule[
        types.State[Optional["TrustRegion.State"], TensorType], Box, ProbabilisticModelType
    ]
):
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

    @overload
    def __init__(
        self: "TrustRegion[ProbabilisticModel]",
        rule: None = None,
        beta: float = 0.7,
        kappa: float = 1e-4,
    ):
        ...

    @overload
    def __init__(
        self: "TrustRegion[ProbabilisticModelType]",
        rule: AcquisitionRule[TensorType, Box, ProbabilisticModelType],
        beta: float = 0.7,
        kappa: float = 1e-4,
    ):
        ...

    def __init__(
        self,
        rule: AcquisitionRule[TensorType, Box, ProbabilisticModelType] | None = None,
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
        models: Mapping[str, ProbabilisticModelType],
        datasets: Optional[Mapping[str, Dataset]] = None,
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
        :param models: The model for each tag.
        :param datasets: The known observer query points and observations. Uses the data for key
            `OBJECTIVE` to calculate the new trust region.
        :return: A function that constructs the next acquisition state and the recommended query
            points from the previous acquisition state.
        :raise KeyError: If ``datasets`` does not contain the key `OBJECTIVE`.
        """
        if datasets is None or OBJECTIVE not in datasets.keys():
            raise ValueError(f"""datasets must be provided and contain the key {OBJECTIVE}""")

        dataset = datasets[OBJECTIVE]

        global_lower = search_space.lower
        global_upper = search_space.upper

        y_min = tf.reduce_min(dataset.observations, axis=0)

        def state_func(
            state: TrustRegion.State | None,
        ) -> tuple[TrustRegion.State | None, TensorType]:

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

            points = self._rule.acquire(acquisition_space, models, datasets=datasets)
            state_ = TrustRegion.State(acquisition_space, eps, y_min, is_global)

            return state_, points

        return state_func
