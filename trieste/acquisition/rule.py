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
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Generic,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)

import numpy as np
from check_shapes import check_shapes, inherit_check_shapes

try:
    import pymoo
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem as PymooProblem
    from pymoo.optimize import minimize
except ImportError:  # pragma: no cover (tested but not by coverage)
    pymoo = None
    PymooProblem = object

import tensorflow as tf

from .. import logging, types
from ..data import Dataset
from ..models import ProbabilisticModel
from ..models.interfaces import (
    HasReparamSampler,
    ModelStack,
    ProbabilisticModelType,
    TrainableSupportsGetKernel,
)
from ..observer import OBJECTIVE
from ..space import (
    Box,
    DiscreteSearchSpace,
    GeneralDiscreteSearchSpace,
    HasOneHotEncoder,
    SearchSpace,
    TaggedMultiSearchSpace,
    TaggedProductSearchSpace,
)
from ..types import State, Tag, TensorType
from ..utils.misc import LocalizedTag
from .function import (
    BatchMonteCarloExpectedImprovement,
    ExpectedImprovement,
    ProbabilityOfImprovement,
)
from .interface import (
    AcquisitionFunction,
    AcquisitionFunctionBuilder,
    GreedyAcquisitionFunctionBuilder,
    SingleModelAcquisitionBuilder,
    SingleModelGreedyAcquisitionBuilder,
    SingleModelVectorizedAcquisitionBuilder,
    VectorizedAcquisitionFunctionBuilder,
)
from .multi_objective import Pareto
from .optimizer import (
    AcquisitionOptimizer,
    automatic_optimizer_selector,
    batchify_joint,
    batchify_vectorize,
)
from .sampler import ExactThompsonSampler, ThompsonSampler
from .utils import get_unique_points_mask, select_nth_output

ResultType = TypeVar("ResultType", covariant=True)
""" Unbound covariant type variable. """

SearchSpaceType = TypeVar("SearchSpaceType", bound=SearchSpace, contravariant=True)
""" Contravariant type variable bound to :class:`~trieste.space.SearchSpace`. """

T = TypeVar("T")
""" Unbound type variable. """


class AcquisitionRule(ABC, Generic[ResultType, SearchSpaceType, ProbabilisticModelType]):
    """
    The central component of the acquisition API.

    An :class:`AcquisitionRule` can produce any value from the search space for this step, and the
    historic data and models. This value is typically a set of query points, either on its own as
    a `TensorType` (see e.g. :class:`EfficientGlobalOptimization`), or within some context
    (see e.g. :class:`BatchTrustRegion`). Indeed, to use an :class:`AcquisitionRule` in the main
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
        models: Mapping[Tag, ProbabilisticModelType],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> ResultType:
        """
        Return a value of type `T_co`. Typically this will be a set of query points, either on its
        own as a `TensorType` (see e.g. :class:`EfficientGlobalOptimization`), or within some
        context (see e.g. :class:`BatchTrustRegion`). We assume that this requires at least models,
        but it may sometimes also need data.

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

    # AcquisitionRule should really have been generic in StateType, but that's too big a change now
    def filter_datasets(
        self, models: Mapping[Tag, ProbabilisticModelType], datasets: Mapping[Tag, Dataset]
    ) -> Mapping[Tag, Dataset] | State[Any | None, Mapping[Tag, Dataset]]:
        """
        Filter the post-acquisition datasets before they are used for model training. For example,
        this can be used to remove points from the post-acquisition datasets that are no longer in
        the search space.
        Rules that need to update their internal state should return a State callable.

        :param models: The model for each tag.
        :param datasets: The updated datasets after previous acquisition step.
        :return: The filtered datasets.
        """
        # No filtering by default.
        return datasets


class LocalDatasetsAcquisitionRule(
    AcquisitionRule[ResultType, SearchSpaceType, ProbabilisticModelType]
):
    """An :class:`AcquisitionRule` that requires local datasets. For example, this is implemented
    by :class:`BatchTrustRegion`."""

    @property
    @abstractmethod
    def num_local_datasets(self) -> int:
        """The number of local datasets required by this rule."""

    @abstractmethod
    def initialize_subspaces(self, search_space: SearchSpaceType) -> None:
        """Create local subspaces for when no initial subspaces are provided."""


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
        initial_acquisition_function: Optional[AcquisitionFunction] = None,
    ): ...

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
        initial_acquisition_function: Optional[AcquisitionFunction] = None,
    ): ...

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
        initial_acquisition_function: Optional[AcquisitionFunction] = None,
    ):
        """
        :param builder: The acquisition function builder to use. Defaults to
            :class:`~trieste.acquisition.ExpectedImprovement`.
        :param optimizer: The optimizer with which to optimize the acquisition function built by
            ``builder``. This should *maximize* the acquisition function, and must be compatible
            with the global search space. Defaults to
            :func:`~trieste.acquisition.optimizer.automatic_optimizer_selector`.
        :param num_query_points: The number of points to acquire.
        :param initial_acquisition_function: The initial acquisition function to use. Defaults
            to using the builder to construct one, but passing in a previously constructed
            function can occasionally be useful (e.g. to preserve random seeds).
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
        self._acquisition_function: Optional[AcquisitionFunction] = initial_acquisition_function

    def __repr__(self) -> str:
        """"""
        return f"""EfficientGlobalOptimization(
        {self._builder!r},
        {self._optimizer!r},
        {self._num_query_points!r})"""

    @property
    def acquisition_function(self) -> Optional[AcquisitionFunction]:
        """The current acquisition function, updated last time :meth:`acquire` was called."""
        return self._acquisition_function

    def acquire(
        self,
        search_space: SearchSpaceType,
        models: Mapping[Tag, ProbabilisticModelType],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
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

        summary_writer = logging.get_tensorboard_writer()
        step_number = logging.get_step_number()
        greedy = isinstance(self._builder, GreedyAcquisitionFunctionBuilder)

        with tf.name_scope("EGO.optimizer" + "[0]" * greedy):
            points = self._optimizer(search_space, self._acquisition_function)

        if summary_writer:
            with summary_writer.as_default(step=step_number):
                batched_points = tf.expand_dims(points, axis=0)
                values = self._acquisition_function(batched_points)[0]
                if len(values) == 1:
                    logging.scalar(
                        "EGO.acquisition_function/maximum_found" + "[0]" * greedy, values[0]
                    )
                else:  # vectorized acquisition function
                    logging.histogram(
                        "EGO.acquisition_function/maximums_found" + "[0]" * greedy, values
                    )

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
                with tf.name_scope(f"EGO.optimizer[{i+1}]"):
                    chosen_point = self._optimizer(search_space, self._acquisition_function)
                points = tf.concat([points, chosen_point], axis=0)

                if summary_writer:
                    with summary_writer.as_default(step=step_number):
                        batched_points = tf.expand_dims(chosen_point, axis=0)
                        values = self._acquisition_function(batched_points)[0]
                        if len(values) == 1:
                            logging.scalar(
                                f"EGO.acquisition_function/maximum_found[{i + 1}]", values[0]
                            )
                        else:  # vectorized acquisition function
                            logging.histogram(
                                f"EGO.acquisition_function/maximums_found[{i+1}]", values
                            )

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

            # since we're compiling, we still need to handle pending_points = [] here
            top = tf.cond(tf.math.greater(1, tf.shape(are_points_equal)[0]), lambda: 0, lambda: 1)

            # this line converts all bool values to 0 and 1
            # then finds first 1 and returns its index as 1x1 tensor
            _, first_index_tensor = tf.math.top_k(tf.cast(are_points_equal, tf.int8), k=top)
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
            message=f"""Point to remove shall be 1xD where D is
                        the last dimension of pending points.
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
    ): ...

    @overload
    def __init__(
        self: "AsynchronousOptimization[SearchSpaceType, ProbabilisticModelType]",
        builder: (
            AcquisitionFunctionBuilder[ProbabilisticModelType]
            | SingleModelAcquisitionBuilder[ProbabilisticModelType]
        ),
        optimizer: AcquisitionOptimizer[SearchSpaceType] | None = None,
        num_query_points: int = 1,
    ): ...

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
        models: Mapping[Tag, ProbabilisticModelType],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
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
            tf.debugging.Assert(self._acquisition_function is not None, [tf.constant([])])

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

            with tf.name_scope("AsynchronousOptimization.optimizer"):
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
        builder: (
            GreedyAcquisitionFunctionBuilder[ProbabilisticModelType]
            | SingleModelGreedyAcquisitionBuilder[ProbabilisticModelType]
        ),
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
        models: Mapping[Tag, ProbabilisticModelType],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
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

            with tf.name_scope("AsynchronousOptimization.optimizer[0]"):
                new_points_batch = self._optimizer(search_space, self._acquisition_function)
            state = state.add_pending_points(new_points_batch)

            summary_writer = logging.get_tensorboard_writer()
            step_number = logging.get_step_number()

            for i in range(self._num_query_points - 1):
                # greedily allocate additional batch elements
                self._acquisition_function = self._builder.update_acquisition_function(
                    self._acquisition_function,
                    models,
                    datasets=datasets,
                    pending_points=state.pending_points,
                    new_optimization_step=False,
                )
                with tf.name_scope(f"AsynchronousOptimization.optimizer[{i+1}]"):
                    new_point = self._optimizer(search_space, self._acquisition_function)
                if summary_writer:
                    with summary_writer.as_default(step=step_number):
                        batched_point = tf.expand_dims(new_point, axis=0)
                        value = self._acquisition_function(batched_point)[0][0]
                        logging.scalar(
                            f"AsyncGreedy.acquisition_function/maximum_found[{i}]", value
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
        models: Mapping[Tag, ProbabilisticModel],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
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
        select_output: Callable[[TensorType], TensorType] = select_nth_output,
    ): ...

    @overload
    def __init__(
        self: "DiscreteThompsonSampling[ProbabilisticModelType]",
        num_search_space_samples: int,
        num_query_points: int,
        thompson_sampler: Optional[ThompsonSampler[ProbabilisticModelType]] = None,
        select_output: Callable[[TensorType], TensorType] = select_nth_output,
    ): ...

    def __init__(
        self,
        num_search_space_samples: int,
        num_query_points: int,
        thompson_sampler: Optional[ThompsonSampler[ProbabilisticModelType]] = None,
        select_output: Callable[[TensorType], TensorType] = select_nth_output,
    ):
        """
        :param num_search_space_samples: The number of points at which to sample the posterior.
        :param num_query_points: The number of points to acquire.
        :param thompson_sampler: Sampler to sample maximisers from the underlying model.
        :param select_output: A method that returns the desired trajectory from a trajectory
            sampler with shape [..., B], where B is a batch dimension. Defaults to the
            :func:~`trieste.acquisition.utils.select_nth_output` function with output dimension 0.
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
        self._select_output = select_output

    def __repr__(self) -> str:
        """"""
        return f"""DiscreteThompsonSampling(
        {self._num_search_space_samples!r},
        {self._num_query_points!r},
        {self._thompson_sampler!r},
        {self._select_output!r})"""

    def acquire(
        self,
        search_space: SearchSpace,
        models: Mapping[Tag, ProbabilisticModelType],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
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
            models[OBJECTIVE],
            self._num_query_points,
            query_points,
            select_output=self._select_output,
        )

        return thompson_samples


class UpdatableSearchSpace(SearchSpace):
    """A search space that can be updated."""

    def __init__(self) -> None:
        self._initialized = False

    @property
    def requires_initialization(self) -> bool:
        """Return `True` if the search space needs to be re-initialized with the latest models
        and datasets, and `False` if it can be just updated."""
        return not self._initialized

    @abstractmethod
    def initialize(
        self,
        models: Optional[Mapping[Tag, ProbabilisticModelType]] = None,
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> None:
        """
        Initialize the search space using the given models and datasets.

        Extending classes must set `self._initialized` to `True` after initialization in this
        method.

        :param models: The model for each tag.
        :param datasets: The dataset for each tag.
        """

    @abstractmethod
    def update(
        self,
        models: Optional[Mapping[Tag, ProbabilisticModelType]] = None,
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> None:
        """
        Update the search space using the given models and datasets.

        :param models: The model for each tag.
        :param datasets: The dataset for each tag.
        """


class UpdatableTrustRegion(UpdatableSearchSpace):
    """An updatable trust region with a concept of a location within a global search space."""

    def __init__(
        self,
        region_index: Optional[int] = None,
        input_active_dims: Optional[Union[slice, Sequence[int]]] = None,
    ) -> None:
        """
        :param region_index: The index of the region in a multi-region search space. This is used to
            identify the local models and datasets to use for acquisition. If `None`, the
            global models and datasets are used.
        :param input_active_dims: The active dimensions of the input space, either a slice or list
            of indices into the columns of the space. If `None`, all dimensions are active.

            When this region is part of a product search-space (via `UpdatableTrustRegionProduct`),
            this is used to select the active dimensions of the full input space that belong to this
            region.
        """
        super().__init__()
        self.region_index = region_index
        self.input_active_dims = input_active_dims

    def _init_location(
        self,
        models: Optional[Mapping[Tag, ProbabilisticModelType]] = None,
        datasets: Optional[Mapping[Tag, Dataset]] = None,
        location_candidate: Optional[TensorType] = None,
    ) -> None:
        """
        Initialize the location of the region, either by sampling a new location from the global
        search space, or by using a candidate location if provided.

        Derived classes can override this method to provide custom initialization logic.

        :param models: The model for each tag.
        :param datasets: The dataset for each tag.
        :param location_candidate: A candidate for the location of the search space. If not
            None, this is used instead of sampling a new location.
        """
        if location_candidate is not None:
            self.location = location_candidate
        else:
            self.location = tf.squeeze(self.global_search_space.sample(1), axis=0)

    @property
    @abstractmethod
    def location(self) -> TensorType:
        """The center of the region."""

    @location.setter
    @abstractmethod
    def location(self, location: TensorType) -> None:
        """Set the center of the region."""

    @property
    @abstractmethod
    def global_search_space(self) -> SearchSpace:
        """The global search space this region lives in."""

    def _get_tags(self, tags: Set[Tag]) -> Tuple[Set[Tag], Set[Tag]]:
        # Separate tags into local (matching index) and global tags (without matching
        # local tag).
        local_gtags = set()  # Set of global part of all local tags.
        global_tags = set()  # Set of all global tags.
        for tag in tags:
            ltag = LocalizedTag.from_tag(tag)
            if not ltag.is_local:
                global_tags.add(tag)
            elif ltag.local_index == self.region_index:
                local_gtags.add(ltag.global_tag)

        # Only keep global tags that don't have a matching local tag.
        global_tags -= local_gtags

        return local_gtags, global_tags

    @overload
    def with_input_active_dims(self, value: TensorType) -> TensorType: ...

    @overload
    def with_input_active_dims(self, value: Dataset) -> Dataset: ...

    @overload
    def with_input_active_dims(self, value: ProbabilisticModel) -> ProbabilisticModel: ...

    def with_input_active_dims(
        self, value: Union[TensorType, Dataset, ProbabilisticModel]
    ) -> Union[TensorType, Dataset, ProbabilisticModel]:
        """
        Select and return active components from the input dimensions of the given value, using
        `input_active_dims` of this search space. If `input_active_dims` is `None`, all dimensions
        are returned.

        For datasets, the active selection is applied to the query points. For models, no
        selection is applied; they are returned as is.

        :param value: The value to select the active input dimensions for.
        :return: The value with the active input dimensions selected.
        """

        # No selection for models.
        # Nothing to do if active dimensions are not set.
        # NOTE: do not replace with isinstance(value, ProbabilisticModel) until
        # https://github.com/secondmind-labs/trieste/issues/836 has been fixed.
        if not isinstance(value, (Dataset, tf.Tensor)) or self.input_active_dims is None:
            return value

        # Select components of query points for datasets.
        if isinstance(value, Dataset):
            input = value.query_points
        else:
            input = value

        if isinstance(self.input_active_dims, slice):
            selected_input = input[..., self.input_active_dims]
        elif self.input_active_dims is not None:
            selected_input = tf.gather(input, self.input_active_dims, axis=-1)

        if isinstance(value, Dataset):
            return Dataset(selected_input, value.observations)
        else:
            return selected_input

    @overload
    def select_in_region(self, mapping: None) -> None: ...

    @overload
    def select_in_region(self, mapping: Mapping[Tag, TensorType]) -> Mapping[Tag, TensorType]: ...

    @overload
    def select_in_region(self, mapping: Mapping[Tag, Dataset]) -> Mapping[Tag, Dataset]: ...

    @overload
    def select_in_region(
        self, mapping: Mapping[Tag, ProbabilisticModel]
    ) -> Mapping[Tag, ProbabilisticModel]: ...

    def select_in_region(
        self, mapping: Optional[Mapping[Tag, Union[TensorType, Dataset, ProbabilisticModel]]]
    ) -> Optional[Mapping[Tag, Union[TensorType, Dataset, ProbabilisticModel]]]:
        """
        Select items belonging to this region for, e.g., acquisition.

        :param mapping: The mapping of items for each tag.
        :return: The items belonging to this region (or `None` if there aren't any).
        """
        if mapping is None:
            _mapping = {}
        elif self.region_index is None:
            # If no index, then return the global items.
            _mapping = {
                tag: self.with_input_active_dims(item)
                for tag, item in mapping.items()
                if not LocalizedTag.from_tag(tag).is_local
            }
        else:
            # Prefer matching local item for each tag, otherwise select the global item.
            local_gtags, global_tags = self._get_tags(set(mapping))

            _mapping = {}
            for tag in local_gtags:
                ltag = LocalizedTag(tag, self.region_index)
                _mapping[ltag] = self.with_input_active_dims(mapping[ltag])
            for tag in global_tags:
                _mapping[tag] = self.with_input_active_dims(mapping[tag])

        return _mapping if _mapping else None

    def get_datasets_filter_mask(
        self, datasets: Optional[Mapping[Tag, Dataset]]
    ) -> Optional[Mapping[Tag, tf.Tensor]]:
        """
        Return a boolean mask that can be used to filter out points from the datasets that
        belong to this region.

        :param datasets: The dataset for each tag.
        :return: A mapping for each tag belonging to this region, to a boolean mask that can be
            used to filter out points from the datasets. A value of `True` indicates that the
            corresponding point should be kept.
        """
        # Only select the region datasets for filtering. Don't directly filter the global dataset.
        assert (
            self.region_index is not None
        ), "the region_index should be set for filtering local datasets"
        if datasets is None:
            return None
        else:
            # Only keep points that are in the region.
            return {
                tag: self.contains(self.with_input_active_dims(dataset.query_points))
                for tag, dataset in datasets.items()
                if LocalizedTag.from_tag(tag).local_index == self.region_index
            }


UpdatableTrustRegionType = TypeVar("UpdatableTrustRegionType", bound=UpdatableTrustRegion)
""" A type variable bound to :class:`UpdatableTrustRegion`. """


@dataclass(frozen=True)
class BatchTrustRegionState(Generic[UpdatableTrustRegionType]):
    """The acquisition state for the :class:`BatchTrustRegion` acquisition rule."""

    subspaces: Sequence[UpdatableTrustRegionType]
    """ The acquisition space's subspaces. """

    subspace_tags: Sequence[str]
    """ The subspaces' tags. """

    def __deepcopy__(
        self, memo: dict[int, object]
    ) -> BatchTrustRegionState[UpdatableTrustRegionType]:
        subspaces_copy = copy.deepcopy(self.subspaces)
        return BatchTrustRegionState(subspaces_copy, self.subspace_tags)

    @property
    def acquisition_space(self) -> TaggedMultiSearchSpace:
        """The acquisition search space."""
        return TaggedMultiSearchSpace(self.subspaces, self.subspace_tags)


class BatchTrustRegion(
    LocalDatasetsAcquisitionRule[
        types.State[Optional[BatchTrustRegionState[UpdatableTrustRegionType]], TensorType],
        SearchSpace,
        ProbabilisticModelType,
    ],
    Generic[ProbabilisticModelType, UpdatableTrustRegionType],
):
    """Abstract class for multi trust region acquisition rules. These are batch algorithms where
    each query point is optimized in parallel, with its own separate trust region.

    Note: to restart or continue an optimization with this rule, either the same instance of the
    rule must be used, or a new instance must be created with the subspaces from a previous
    state. This is because the internal state of the rule cannot be restored directly from a state
    object.
    """

    def __init__(
        self: "BatchTrustRegion[ProbabilisticModelType, UpdatableTrustRegionType]",
        init_subspaces: Union[
            None, UpdatableTrustRegionType, Sequence[UpdatableTrustRegionType]
        ] = None,
        rule: AcquisitionRule[TensorType, SearchSpace, ProbabilisticModelType] | None = None,
    ):
        """
        :param init_subspaces: The initial search spaces for each trust region. If `None`, default
            subspaces of type :class:`UpdatableTrustRegionType` will be created, with length
            equal to the number of query points in the base `rule`.
        :param rule: The acquisition rule that defines how to search for a new query point in each
            subspace.

            If `None`, defaults to :class:`~trieste.acquisition.DiscreteThompsonSampling` with
            a batch size of 1 for `TURBOBox` subspaces, and
            :class:`~trieste.acquisition.EfficientGlobalOptimization` otherwise.
        """
        # If init_subspaces are not provided, leave it to the subclasses to create them.
        self._init_subspaces = None
        self._tags = None
        if init_subspaces is not None:
            if not isinstance(init_subspaces, Sequence):
                init_subspaces = [init_subspaces]
            self._init_subspaces = tuple(init_subspaces)
            for index, subspace in enumerate(self._init_subspaces):
                subspace.region_index = index  # Override the index.
            self._tags = tuple(str(index) for index, _ in enumerate(self._init_subspaces))

        self._rule = rule
        # The rules for each subspace. These are only used when we want to run the base rule
        # sequentially for each subspace. Theses are set in `acquire`.
        self._rules: Optional[
            Sequence[AcquisitionRule[TensorType, SearchSpace, ProbabilisticModelType]]
        ] = None

    def __repr__(self) -> str:
        """"""
        return f"""{self.__class__.__name__}({self._init_subspaces!r}, {self._rule!r})"""

    @property
    def num_local_datasets(self) -> int:
        assert self._init_subspaces is not None, "the subspaces have not been initialized"
        return len(self._init_subspaces)

    def acquire(
        self,
        search_space: SearchSpace,
        models: Mapping[Tag, ProbabilisticModelType],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> types.State[BatchTrustRegionState[UpdatableTrustRegionType] | None, TensorType]:
        """
        Use the ``rule`` specified at :meth:`~BatchTrustRegion.__init__` to find new
        query points. Return a function that constructs these points given a previous trust region
        state.

        If state is None, initialize the subspaces by picking new locations. Otherwise,
        update the existing subspaces.

        Re-initialize the subspaces if necessary, potentially looking at the entire group.

        :param search_space: The acquisition search space for *this step*.
        :param models: The model for each tag.
        :param datasets: The known observer query points and observations for each tag.
        :return: A function that constructs the next acquisition state and the recommended query
            points from the previous acquisition state.
        """

        # initialize subspaces
        self.initialize_subspaces(search_space)

        # Subspaces should be initialised by the time we call `acquire`.
        assert self._tags is not None
        assert self._init_subspaces is not None

        # Implement heuristic defaults for the rule if not specified by the user.
        if self._rule is None:
            # Use first subspace to determine the type of the base rule.
            if isinstance(self._init_subspaces[0], TURBOBox):
                # Default to Thompson sampling with batches of size 1.
                self._rule = DiscreteThompsonSampling(
                    tf.minimum(100 * search_space.dimension, 5_000), 1
                )
            else:
                self._rule = EfficientGlobalOptimization()

        num_local_models = Counter(
            LocalizedTag.from_tag(tag).global_tag
            for tag in models
            if LocalizedTag.from_tag(tag).is_local
        )
        num_local_models_vals = set(num_local_models.values())
        assert (
            len(num_local_models_vals) <= 1
        ), f"The number of local models should be the same for all tags, got {num_local_models}"
        _num_local_models = sum(num_local_models_vals)

        num_subspaces = len(self._tags)
        assert _num_local_models in [0, num_subspaces], (
            f"When using local models, the number of subspaces {num_subspaces} should be equal to "
            f"the number of local models {_num_local_models}"
        )

        # If we have local models or not using a base-rule that supports batched acquisition,
        # run the (deepcopied) base rule sequentially for each subspace. Note: we only support
        # batching for EfficientGlobalOptimization.
        # Otherwise, run the base rule as is (i.e as a batch), once with all models and datasets.
        # Note: this should only trigger on the first call to `acquire`, as after that we will
        # have a list of rules in `self._rules`.
        if self._rules is None and not (
            _num_local_models == 0 and isinstance(self._rule, EfficientGlobalOptimization)
        ):
            self._rules = [copy.deepcopy(self._rule) for _ in range(num_subspaces)]

        def state_func(
            state: BatchTrustRegionState[UpdatableTrustRegionType] | None,
        ) -> Tuple[BatchTrustRegionState[UpdatableTrustRegionType] | None, TensorType]:
            # Check again to keep mypy happy.
            assert self._tags is not None
            assert self._init_subspaces is not None
            assert self._rule is not None

            # If state is set, the tags should be the same as the tags of the initial space.
            if state is not None:
                assert self._tags == tuple(state.subspace_tags), (
                    f"The tags of the state acquisition space "
                    f"{state.subspace_tags} should be the same as the tags of the "
                    f"BatchTrustRegion acquisition rule {self._tags}"
                )
                assert len(state.subspaces) == len(state.subspace_tags), (
                    f"Inconsistent number of subspaces: {len(state.subspaces)} subspaces"
                    f"and {len(state.subspace_tags)} tags"
                )
                subspaces = state.subspaces
            else:
                subspaces = self._init_subspaces

            # If the base rule is a sequence, run it sequentially for each subspace.
            # See earlier comments.
            if self._rules is not None:
                _points = []
                for tag, subspace, rule in zip(self._tags, subspaces, self._rules):
                    _models = subspace.select_in_region(models)
                    _datasets = subspace.select_in_region(datasets)
                    assert _models is not None
                    # Remap all local tags to global ones. One reason is that single model
                    # acquisition builders expect OBJECTIVE to exist.
                    _models = {
                        LocalizedTag.from_tag(tag).global_tag: model
                        for tag, model in _models.items()
                    }
                    if _datasets is not None:
                        _datasets = {
                            LocalizedTag.from_tag(tag).global_tag: dataset
                            for tag, dataset in _datasets.items()
                        }
                    _points.append(rule.acquire(subspace, _models, _datasets))
                points = tf.stack(_points, axis=1)
            else:
                # Filter out local datasets as this is a rule (currently only EGO) with normal
                # acquisition functions that don't expect local datasets.
                # Note: no need to filter out local models, as setups with local models
                # are handled above (i.e. we run the base rule sequentially for each subspace).
                if datasets is not None:
                    _datasets = {
                        tag: dataset
                        for tag, dataset in datasets.items()
                        if not LocalizedTag.from_tag(tag).is_local
                    }
                else:
                    _datasets = None
                acquisition_space = TaggedMultiSearchSpace(subspaces, self._tags)
                points = self._rule.acquire(acquisition_space, models, _datasets)

            state_ = BatchTrustRegionState(subspaces, self._tags)
            return state_, tf.reshape(points, [-1, len(subspaces), points.shape[-1]])

        return state_func

    def maybe_initialize_subspaces(
        self,
        subspaces: Sequence[UpdatableTrustRegionType],
        models: Mapping[Tag, ProbabilisticModelType],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> None:
        """
        Initialize subspaces if necessary.
        Get a mask of subspaces that need to be initialized using an abstract method.
        Initialize individual subpaces by calling the method of the UpdatableTrustRegionType class.

        This method can be overridden by subclasses to change this behaviour.
        """
        mask = self.get_initialize_subspaces_mask(subspaces, models, datasets)
        tf.debugging.assert_equal(
            tf.shape(mask),
            (len(subspaces),),
            message="The mask for initializing subspaces should be of the same length as the "
            "number of subspaces",
        )
        for ix, subspace in enumerate(subspaces):
            if mask[ix]:
                subspace.initialize(models, datasets)

    @abstractmethod
    @check_shapes("return: [V]")
    def get_initialize_subspaces_mask(
        self,
        subspaces: Sequence[UpdatableTrustRegionType],
        models: Mapping[Tag, ProbabilisticModelType],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> TensorType:
        """
        Return a boolean mask for subspaces that should be initialized.
        This method is called during the acquisition step to determine which subspaces should be
        initialized and which should be updated. The subspaces corresponding to True values in the
        mask will be re-initialized.

        :param subspaces: The sequence of subspaces.
        :param models: The model for each tag.
        :param datasets: The dataset for each tag.
        :return: A boolean mask of length V, where V is the number of subspaces.
        """

    def filter_datasets(
        self, models: Mapping[Tag, ProbabilisticModelType], datasets: Mapping[Tag, Dataset]
    ) -> types.State[BatchTrustRegionState[UpdatableTrustRegionType] | None, Mapping[Tag, Dataset]]:
        def state_func(
            state: BatchTrustRegionState[UpdatableTrustRegionType] | None,
        ) -> Tuple[BatchTrustRegionState[UpdatableTrustRegionType] | None, Mapping[Tag, Dataset]]:
            if state is not None:
                assert self._tags == tuple(state.subspace_tags), (
                    f"The tags of the state acquisition space "
                    f"{state.subspace_tags} should be the same as the tags of the "
                    f"BatchTrustRegion acquisition rule {self._tags}"
                )
                assert len(state.subspaces) == len(state.subspace_tags), (
                    f"Inconsistent number of subspaces: {len(state.subspaces)} subspaces"
                    f"and {len(state.subspace_tags)} tags"
                )
                subspaces = tuple(state.subspaces)
            else:
                assert self._init_subspaces is not None, "the subspaces have not been initialized"
                assert self._tags is not None
                subspaces = self._init_subspaces

            # make a deepcopy to avoid modifying any user copies
            subspaces = copy.deepcopy(subspaces)

            # Update subspaces with the latest datasets.
            for subspace in subspaces:
                # Re-initialize or update the subspace, depending on the property.
                if subspace.requires_initialization:
                    subspace.initialize(models, datasets)
                else:
                    subspace.update(models, datasets)
            self.maybe_initialize_subspaces(subspaces, models, datasets)

            # Filter out points that are not in any of the subspaces. This is done by creating a
            # mask for each local dataset that is True for points that are in any subspace.
            used_masks = {
                tag: tf.zeros(dataset.query_points.shape[:-1], dtype=tf.bool)
                for tag, dataset in datasets.items()
                if LocalizedTag.from_tag(tag).is_local
            }

            for subspace in subspaces:
                in_region_masks = subspace.get_datasets_filter_mask(datasets)
                if in_region_masks is not None:
                    for tag, in_region in in_region_masks.items():
                        ltag = LocalizedTag.from_tag(tag)
                        assert ltag.is_local, f"can only filter local tags, got {tag}"
                        used_masks[tag] = tf.logical_or(used_masks[tag], in_region)

            filtered_datasets = {}
            for tag, used_mask in used_masks.items():
                filtered_datasets[tag] = Dataset(
                    tf.boolean_mask(datasets[tag].query_points, used_mask),
                    tf.boolean_mask(datasets[tag].observations, used_mask),
                )

            # Include global datasets unmodified.
            for tag, dataset in datasets.items():
                if not LocalizedTag.from_tag(tag).is_local:
                    filtered_datasets[tag] = dataset

            state_ = BatchTrustRegionState(subspaces, self._tags)
            return state_, filtered_datasets

        return state_func


class HypercubeTrustRegion(UpdatableTrustRegion):
    """
    An abstract updatable trust region that defines a hypercube region in the global search space.
    The region is defined by a location and a size in each dimension. This class is used to
    implement different types of search spaces, e.g. continuous (SingleObjectiveTrustRegionBox) and
    discrete (SingleObjectiveTrustRegionDiscrete).

    Derived classes must implement the `_update_domain` method to update the domain of the region
    based on the location and size.

    In the default implementation, the region is updated based on the minimum observed value in
    the region from a single objective dataset. The region is expanded if the minimum is improved,
    and contracted otherwise. Derived classes can override how this minimum is calculated, e.g. by
    utilizing multiple datasets.
    """

    def __init__(
        self,
        beta: float = 0.7,
        kappa: float = 1e-4,
        zeta: float = 0.5,
        min_eps: float = 1e-2,
    ):
        """
        Calculates the bounds of the region from the location/center and global bounds.

        :param beta: The inverse of the trust region contraction factor.
        :param kappa: Scales the threshold for the minimal improvement required for a step to be
            considered a success.
        :param zeta: The initial size of the trust region is ``zeta`` times the size of the global
            search space.
        :param min_eps: The minimal size of the search space. If the size of the search space is
            smaller than this, the search space is reinitialized.
        """
        self._beta = beta
        self._kappa = kappa
        self._zeta = zeta
        self._min_eps = min_eps
        self._step_is_success = False
        # Randomly pick initial value of `self.location` for setting the bounds from the
        # global search space.
        self._init_location()
        self._init_eps()
        self._update_domain()
        # Initial value of the region minimum is set to infinity as we have not yet observed any
        # data.
        self._y_min = tf.constant(np.inf, dtype=self.location.dtype)

    def _init_eps(self) -> None:
        if not isinstance(self.global_search_space, HasOneHotEncoder):
            self.eps = self._zeta * (
                self.global_search_space.upper - self.global_search_space.lower
            )
        else:
            # categorical space distance is hardcoded to a (Hamming) distance of 1
            self.eps = 1

    @abstractmethod
    def _update_domain(self) -> None:
        """Update the local domain of the region."""
        ...

    @property
    def requires_initialization(self) -> bool:
        """
        Return `True` if the search space needs to be initialized, and `False` otherwise.

        If uninitialized, or the size of the region is less than the minimum size, re-initialize
        the region.
        """
        return not self._initialized or tf.reduce_any(self.eps < self._min_eps)

    def initialize(
        self,
        models: Optional[Mapping[Tag, ProbabilisticModelType]] = None,
        datasets: Optional[Mapping[Tag, Dataset]] = None,
        location_candidate: Optional[TensorType] = None,
    ) -> None:
        """
        Initialize the region by sampling a location from the global search space and setting the
        local region bounds around it.

        :param models: The model for each tag.
        :param datasets: The dataset for each tag.
        :param location_candidate: A candidate for the location of the search space. If not
            None, this is used instead of sampling a new location.
        """

        # This returns datasets that are relevant for this region only, for each tag present in the
        # dataset. Per tag it return a local dataset with a matching region index where it can;
        # failing that it will return the global dataset for that tag.
        datasets = self.select_in_region(datasets)
        self._init_location(models, datasets, location_candidate=location_candidate)
        self._step_is_success = False
        self._init_eps()
        self._update_domain()
        # We haven't necessarily observed any data yet for this region; force first step to always
        # be successful by setting the minimum to infinity.
        self._y_min = tf.constant(np.inf, dtype=self.location.dtype)
        self._initialized = True

    def update(
        self,
        models: Optional[Mapping[Tag, ProbabilisticModelType]] = None,
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> None:
        """
        Update this region, including center/location, using the given dataset.

        If the new optimum improves over the previous optimum by some threshold (that scales
        linearly with ``kappa``), the previous acquisition is considered successful.

        If the previous acquisition was successful, the size is increased by a factor
        ``1 / beta``. Conversely, if it was unsuccessful, the size is reduced by the factor
        ``beta``.

        :param models: The model for each tag.
        :param datasets: The dataset for each tag.
        """
        datasets = self.select_in_region(datasets)  # See `select_in_region` comment above.
        x_min, y_min = self.get_dataset_min(datasets)

        if isinstance(self.global_search_space, HasOneHotEncoder):
            self._step_is_success = y_min < self._y_min
        else:
            tr_volume = tf.reduce_prod(self.upper - self.lower)
            self._step_is_success = y_min < self._y_min - self._kappa * tr_volume
            self.eps = self.eps / self._beta if self._step_is_success else self.eps * self._beta

        # Only update the location if the step was successful.
        if self._step_is_success:
            self.location = x_min
            self._y_min = y_min

        self._update_domain()

    @check_shapes(
        "query_points: [N, D]",
        "values: [N, 1]",
        "return[0]: [D]",
        "return[1]: []",
    )
    def get_values_min(
        self,
        query_points: TensorType,
        values: TensorType,
        num_query_points: Optional[int] = None,
        in_region_only: bool = True,
    ) -> Tuple[TensorType, TensorType]:
        """
        Calculate the minimum of the region over the given values, returning the query point and
        value of the minimum. Optionally, only consider query points that are contained in the
        region.

        :param query_points: The query points corresponding to the values.
        :param values: The values to find the minimum over.
        :param num_query_points: The number of latest query points to use for calculating the
            minimum. If None, all query points are used.
        :param in_region_only: If True, only consider points contained in the region.
        :return: The query point and value of the minimum.
        """

        if num_query_points is None:
            qps = query_points
        else:
            qps = query_points[-num_query_points:]

        if in_region_only:
            in_tr = self.contains(qps)  # Only consider points in the region.
            in_tr_obs = tf.where(
                tf.expand_dims(in_tr, axis=-1),
                values,
                tf.constant(np.inf, dtype=values.dtype),
            )
        else:
            in_tr_obs = values  # Consider all points.

        ix = tf.argmin(in_tr_obs)
        x_min = tf.gather(qps, ix)
        y_min = tf.gather(in_tr_obs, ix)

        return tf.squeeze(x_min, axis=0), tf.squeeze(y_min)

    @check_shapes(
        "return[0]: [D]",
        "return[1]: []",
    )
    def get_dataset_min(
        self, datasets: Optional[Mapping[Tag, Dataset]]
    ) -> Tuple[TensorType, TensorType]:
        """
        Calculate the minimum of the region using the given dataset, returning the query point and
        value of the minimum.

        The default implementation supports a single objective dataset only. This can be
        overridden by subclasses to support multiple datasets.

        :param datasets: The datasets to use for finding the minimum.
        :return: The query point and value of the minimum.
        """

        if (
            datasets is None
            or len(datasets) != 1
            or LocalizedTag.from_tag(next(iter(datasets))).global_tag != OBJECTIVE
        ):
            raise ValueError("""a single OBJECTIVE dataset must be provided""")
        dataset = next(iter(datasets.values()))
        return self.get_values_min(dataset.query_points, dataset.observations, in_region_only=True)


class UpdatableTrustRegionBox(Box, UpdatableTrustRegion):
    """
    A simple updatable box search space with a center location and an associated global search
    space.
    """

    def __init__(
        self,
        global_search_space: Box,
        region_index: Optional[int] = None,
        input_active_dims: Optional[Union[slice, Sequence[int]]] = None,
    ):
        """
        :param global_search_space: The global search space this search space lives in.
        :param region_index: The index of the region in a multi-region search space. This is used to
            identify the local models and datasets to use for acquisition. If `None`, the
            global models and datasets are used.
        :param input_active_dims: The active dimensions of the input space, either a slice or list
            of indices into the columns of the space. If `None`, all dimensions are active.
        """
        Box.__init__(self, global_search_space.lower, global_search_space.upper)
        UpdatableTrustRegion.__init__(self, region_index, input_active_dims)
        self._global_search_space = global_search_space

    @property
    def location(self) -> TensorType:
        return self._location

    @location.setter
    def location(self, location: TensorType) -> None:
        self._location = location

    @property
    def global_search_space(self) -> Box:
        return self._global_search_space

    def _get_bounds_within_distance(self, eps: TensorType) -> Tuple[TensorType, TensorType]:
        # Helper method to get the box bounds within a distance `eps` of the region location.
        lower = tf.reduce_max([self.global_search_space.lower, self.location - eps], axis=0)
        upper = tf.reduce_min([self.global_search_space.upper, self.location + eps], axis=0)
        return lower, upper


class SingleObjectiveTrustRegionBox(UpdatableTrustRegionBox, HypercubeTrustRegion):
    """
    An updatable continuous trust region that defines a box region in the global search space.
    The region is updated based on the best point found in the region.
    """

    def __init__(
        self,
        global_search_space: Box,
        beta: float = 0.7,
        kappa: float = 1e-4,
        zeta: float = 0.5,
        min_eps: float = 1e-2,
        region_index: Optional[int] = None,
        input_active_dims: Optional[Union[slice, Sequence[int]]] = None,
    ):
        """
        Calculates the bounds of the box from the location/center and global bounds.

        :param global_search_space: The global search space this search space lives in.
        :param beta: The inverse of the trust region contraction factor.
        :param kappa: Scales the threshold for the minimal improvement required for a step to be
            considered a success.
        :param zeta: The initial size of the trust region is ``zeta`` times the size of the global
            search space.
        :param min_eps: The minimal size of the search space. If the size of the search space is
            smaller than this, the search space is reinitialized.
        :param region_index: The index of the region in a multi-region search space. This is used to
            identify the local models and datasets to use for acquisition. If `None`, the
            global models and datasets are used.
        :param input_active_dims: The active dimensions of the input space, either a slice or list
            of indices into the columns of the space. If `None`, all dimensions are active.
        """
        UpdatableTrustRegionBox.__init__(self, global_search_space, region_index, input_active_dims)
        HypercubeTrustRegion.__init__(self, beta, kappa, zeta, min_eps)

    def _update_domain(self) -> None:
        self._lower, self._upper = self._get_bounds_within_distance(self.eps)


class BatchTrustRegionBox(BatchTrustRegion[ProbabilisticModelType, UpdatableTrustRegionBox]):
    """
    Implements the :class:`BatchTrustRegion` *trust region* acquisition rule for box regions.
    This is intended to be used for single-objective optimization with batching.
    """

    def initialize_subspaces(self, search_space: SearchSpace) -> None:
        # If no initial subspaces were provided, create N default subspaces, where N is the
        # number of query points in the base-rule.
        # Currently the detection for N is only implemented for EGO.
        # Note: the reason we don't create the default subspaces in `__init__` is because we
        # don't have the global search space at that point.
        if self._init_subspaces is None:
            if isinstance(self._rule, EfficientGlobalOptimization):
                num_query_points = self._rule._num_query_points
            else:
                num_query_points = 1

            assert isinstance(
                search_space, Box
            ), f"search space should be a Box, got {type(search_space)}"
            init_subspaces: Tuple[UpdatableTrustRegionBox, ...] = tuple(
                SingleObjectiveTrustRegionBox(search_space) for _ in range(num_query_points)
            )
            self._init_subspaces = init_subspaces
            for index, subspace in enumerate(self._init_subspaces):
                subspace.region_index = index  # Override the index.
            self._tags = tuple(str(index) for index in range(self.num_local_datasets))

    def acquire(
        self,
        search_space: SearchSpace,
        models: Mapping[Tag, ProbabilisticModelType],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> types.State[BatchTrustRegionState[UpdatableTrustRegionBox] | None, TensorType]:
        # Ensure passed in global search space is always the same as the search space passed to
        # the subspaces.
        if self._init_subspaces is not None:
            for subspace in self._init_subspaces:
                assert subspace.global_search_space == search_space, (
                    "The global search space of the subspaces should be the same as the "
                    "search space passed to the BatchTrustRegionBox acquisition rule. "
                    "If you want to change the global search space, you should recreate the rule. "
                    "Note: all subspaces should be initialized with the same global search space."
                )

        return super().acquire(search_space, models, datasets)

    @inherit_check_shapes
    def get_initialize_subspaces_mask(
        self,
        subspaces: Sequence[UpdatableTrustRegionBox],
        models: Mapping[Tag, ProbabilisticModelType],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> TensorType:
        # Initialize the subspaces that have non-unique locations.
        centres = tf.stack([subspace.location for subspace in subspaces])
        return tf.logical_not(get_unique_points_mask(centres, tolerance=1e-6))


class TREGOBox(SingleObjectiveTrustRegionBox):
    """
    A box trust region algorithm that alternates between regular EGO steps and local steps within a
    trust region. See :cite:`diouane2022trego` for details.

    At construction, starts in global mode using ``global_search_space`` as the search space
    for the first step. Subsequent re-initializations use the trust region as the search space for
    the next step.

    If the previous acquisition was successful, ``global_search_space`` is used as the new
    search space. If the previous step was unsuccessful, the search space is changed to the
    trust region if it was global, and vice versa.

    If the previous acquisition was over the trust region, the size of the trust region is
    modified.

    **Note:** The acquisition search space will never extend beyond the boundary of the
    ``global_search_space``. For a local search, the actual search space will be the
    intersection of the trust region and ``global_search_space``.
    """

    def __init__(
        self,
        global_search_space: Box,
        beta: float = 0.7,
        kappa: float = 1e-4,
        zeta: float = 0.5,
        min_eps: float = 1e-2,
        region_index: Optional[int] = None,
        input_active_dims: Optional[Union[slice, Sequence[int]]] = None,
    ):
        self._is_global = False
        super().__init__(
            global_search_space=global_search_space,
            beta=beta,
            kappa=kappa,
            zeta=zeta,
            min_eps=min_eps,
            region_index=region_index,
            input_active_dims=input_active_dims,
        )

    @property
    def eps(self) -> TensorType:
        """The size of the search space."""
        return self._eps

    @eps.setter
    def eps(self, eps: TensorType) -> None:
        """Set the size of the search space."""
        # Don't change the eps in global mode.
        if not self._is_global:
            self._eps = eps

    def _update_domain(self) -> None:
        self._is_global = self._step_is_success or not self._is_global

        # Use global bounds in global mode.
        if self._is_global:
            self._lower = self.global_search_space.lower
            self._upper = self.global_search_space.upper
        else:
            super()._update_domain()

    def initialize(
        self,
        models: Optional[Mapping[Tag, ProbabilisticModelType]] = None,
        datasets: Optional[Mapping[Tag, Dataset]] = None,
        location_candidate: Optional[TensorType] = None,
    ) -> None:
        # `True` if the search space is global, else `False` if it is local.
        # May be a scalar boolean `TensorType` instead of a `bool`.
        #
        # Start in global mode at construction time. Use local mode for subsequent
        # re-initializations. Note the calls to `_update_domain` switch the mode, so the values
        # here are inverted.
        self._is_global = self._initialized

        super().initialize(models, datasets, location_candidate=location_candidate)

    def get_datasets_filter_mask(
        self, datasets: Optional[Mapping[Tag, Dataset]]
    ) -> Optional[Mapping[Tag, tf.Tensor]]:
        # Only select the region datasets for filtering. Don't directly filter the global dataset.
        assert (
            self.region_index is not None
        ), "the region_index should be set for filtering local datasets"
        if datasets is None:
            return None
        else:
            # Don't filter out any points from the dataset. Always keep the entire dataset.
            return {
                tag: tf.ones(
                    tf.shape(self.with_input_active_dims(dataset.query_points))[:-1], dtype=tf.bool
                )
                for tag, dataset in datasets.items()
                if LocalizedTag.from_tag(tag).local_index == self.region_index
            }

    @inherit_check_shapes
    def get_dataset_min(
        self, datasets: Optional[Mapping[Tag, Dataset]]
    ) -> Tuple[TensorType, TensorType]:
        if (
            datasets is None
            or len(datasets) != 1
            or LocalizedTag.from_tag(next(iter(datasets))).global_tag != OBJECTIVE
        ):
            raise ValueError("""a single OBJECTIVE dataset must be provided""")
        dataset = next(iter(datasets.values()))

        # Always return the global minimum.
        return self.get_values_min(dataset.query_points, dataset.observations, in_region_only=False)


class TURBOBox(UpdatableTrustRegionBox):
    """Implements the TURBO algorithm as detailed in :cite:`eriksson2019scalable`."""

    def __init__(
        self,
        global_search_space: Box,
        L_min: Optional[float] = None,
        L_init: Optional[float] = None,
        L_max: Optional[float] = None,
        success_tolerance: int = 3,
        failure_tolerance: Optional[int] = None,
        region_index: Optional[int] = None,
        input_active_dims: Optional[Union[slice, Sequence[int]]] = None,
    ):
        """
        Note that the optional parameters are set by a heuristic if not given by the user.

        :param global_search_space: The global search space.
        :param L_min: Minimum allowed length of the trust region.
        :param L_init: Initial length of the trust region.
        :param L_max: Maximum allowed length of the trust region.
        :param success_tolerance: Number of consecutive successes before changing region size.
        :param failure tolerance: Number of consecutive failures before changing region size.
        :param region_index: The index of the region in a multi-region search space. This is used to
            identify the local models and datasets to use for acquisition. If `None`, the
            global models and datasets are used.
        :param input_active_dims: The active dimensions of the input space, either a slice or list
            of indices into the columns of the space. If `None`, all dimensions are active.
        """
        super().__init__(global_search_space, region_index, input_active_dims)
        self._init_location()

        search_space_max_width = tf.reduce_max(
            global_search_space.upper - global_search_space.lower
        )
        if L_min is None:
            L_min = (0.5**7) * search_space_max_width
        if L_init is None:
            L_init = 0.8 * search_space_max_width
        if L_max is None:
            L_max = 1.6 * search_space_max_width

        if L_min <= 0:
            raise ValueError(f"L_min must be postive, got {L_min}")
        if L_init <= 0:
            raise ValueError(f"L_init must be postive, got {L_init}")
        if L_max <= 0:
            raise ValueError(f"L_max must be postive, got {L_max}")

        self.L_min = L_min
        self.L_init = L_init
        self.L_max = L_max
        self.L = L_init
        self.success_tolerance = success_tolerance
        self.failure_tolerance = (
            failure_tolerance if failure_tolerance is not None else global_search_space.dimension
        )

        self.success_counter = 0
        self.failure_counter = 0

        if self.success_tolerance <= 0:
            raise ValueError(
                f"success tolerance must be an integer greater than 0, got {self.success_tolerance}"
            )
        if self.failure_tolerance <= 0:
            raise ValueError(
                f"success tolerance must be an integer greater than 0, got {self.failure_tolerance}"
            )

        self.y_min = np.inf
        # Initialise to the full global search space size.
        self.tr_width = global_search_space.upper - global_search_space.lower
        self._update_domain()

    def _set_tr_width(self, models: Optional[Mapping[Tag, ProbabilisticModelType]] = None) -> None:
        # Set the width of the trust region based on the local model.
        if (
            models is None
            or len(models) != 1
            or LocalizedTag.from_tag(next(iter(models))).global_tag != OBJECTIVE
        ):
            raise ValueError("""a single OBJECTIVE model must be provided""")
        model = next(iter(models.values()))
        assert isinstance(
            model, TrainableSupportsGetKernel
        ), f"the model should be of type TrainableSupportsGetKernel, got {type(model)}"

        lengthscales = (
            model.get_kernel().lengthscales
        )  # stretch region according to model lengthscales

        # Select the input lengthscales that are active for this region.
        if tf.size(lengthscales) > 1:
            lengthscales = self.with_input_active_dims(tf.convert_to_tensor(lengthscales))

        self.tr_width = (
            lengthscales
            * self.L
            / tf.reduce_prod(lengthscales) ** (1.0 / self.global_search_space.lower.shape[-1])
        )  # keep volume fixed

    def _update_domain(self) -> None:
        self._lower = tf.reduce_max(
            [self.global_search_space.lower, self.location - self.tr_width / 2.0], axis=0
        )
        self._upper = tf.reduce_min(
            [self.global_search_space.upper, self.location + self.tr_width / 2.0], axis=0
        )

    def initialize(
        self,
        models: Optional[Mapping[Tag, ProbabilisticModelType]] = None,
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> None:
        # Use the full dataset to determine the best point.
        datasets = self.select_in_region(datasets)
        x_min, self.y_min = self.get_dataset_min(datasets)
        self.location: TensorType = x_min

        self.L, self.failure_counter, self.success_counter = self.L_init, 0, 0

        models = self.select_in_region(models)
        self._set_tr_width(models)
        self._update_domain()
        self._initialized = True

    def update(
        self,
        models: Optional[Mapping[Tag, ProbabilisticModelType]] = None,
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> None:
        # Use the full dataset to determine the best point.
        datasets = self.select_in_region(datasets)
        x_min, y_min = self.get_dataset_min(datasets)
        self.location = x_min

        step_is_success = y_min < self.y_min - 1e-10  # maybe make this stronger?
        self.y_min = y_min
        self.failure_counter = (
            0 if step_is_success else self.failure_counter + 1
        )  # update or reset counter
        self.success_counter = (
            self.success_counter + 1 if step_is_success else 0
        )  # update or reset counter
        if self.success_counter == self.success_tolerance:
            self.L *= 2.0  # make region bigger
            self.success_counter = 0
        elif self.failure_counter == self.failure_tolerance:
            self.L *= 0.5  # make region smaller
            self.failure_counter = 0

        self.L = tf.minimum(self.L, self.L_max)
        if self.L < self.L_min:  # if gets too small then start again
            self.L, self.failure_counter, self.success_counter = self.L_init, 0, 0

        models = self.select_in_region(models)
        self._set_tr_width(models)
        self._update_domain()

    @check_shapes(
        "return[0]: [D]",
        "return[1]: []",
    )
    def get_dataset_min(
        self, datasets: Optional[Mapping[Tag, Dataset]]
    ) -> Tuple[TensorType, TensorType]:
        """Calculate the minimum of the box using the given dataset."""
        if (
            datasets is None
            or len(datasets) != 1
            or LocalizedTag.from_tag(next(iter(datasets))).global_tag != OBJECTIVE
        ):
            raise ValueError("""a single OBJECTIVE dataset must be provided""")
        dataset = next(iter(datasets.values()))

        ix = tf.argmin(dataset.observations)
        x_min = tf.gather(dataset.query_points, ix)
        y_min = tf.gather(dataset.observations, ix)

        return tf.squeeze(x_min, axis=0), tf.squeeze(y_min)


class UpdatableTrustRegionDiscrete(DiscreteSearchSpace, UpdatableTrustRegion):
    """
    An updatable discrete search space with an associated global search space.
    """

    def __init__(
        self,
        global_search_space: GeneralDiscreteSearchSpace,
        region_index: Optional[int] = None,
        input_active_dims: Optional[Union[slice, Sequence[int]]] = None,
    ):
        """
        :param global_search_space: The global search space this search space lives in.
        :param region_index: The index of the region in a multi-region search space. This is used to
            identify the local models and datasets to use for acquisition. If `None`, the
            global models and datasets are used.
        :param input_active_dims: The active dimensions of the input space, either a slice or list
            of indices into the columns of the space. If `None`, all dimensions are active.
        """
        # Ensure global_points is a copied tensor, in case a variable is passed in.
        DiscreteSearchSpace.__init__(self, tf.constant(global_search_space.points))
        UpdatableTrustRegion.__init__(self, region_index, input_active_dims)
        self._global_search_space = global_search_space

    @property
    def location(self) -> TensorType:
        return tf.reshape(tf.gather(self.global_search_space.points, self._location_ix), (-1,))

    @location.setter
    def location(self, location: TensorType) -> None:
        # Keep the index of the location in the global search space, instead of directly storing
        # the location.
        location_ix = tf.where(tf.reduce_all(self.global_search_space.points == location, axis=-1))
        if tf.size(location_ix) == 0:
            raise ValueError(f"location {location} not found in the global search space")
        self._location_ix = tf.squeeze(location_ix, axis=-1)

    @property
    def global_search_space(self) -> GeneralDiscreteSearchSpace:
        return self._global_search_space

    def _compute_global_distances(self) -> TensorType:
        # Helper method to compute and return pairwise distances along each axis in the
        # global search space.

        points = self.global_search_space.points
        if isinstance(self.global_search_space, HasOneHotEncoder):
            # use Hamming distance for categorical spaces
            return tf.math.reduce_sum(
                tf.where(tf.expand_dims(points, -2) == tf.expand_dims(points, -3), 0, 1),
                axis=-1,
                keepdims=True,  # (keep last dim for reduce_all distance calculation below)
            )  # [num_points, num_points, 1]
        else:
            return tf.abs(
                tf.expand_dims(points, -2) - tf.expand_dims(points, -3)
            )  # [num_points, num_points, D]

    def _get_points_within_distance(
        self, global_distances: TensorType, distance: TensorType
    ) -> TensorType:
        # Helper method to return subset of global points within a given distance of the
        # region location. Takes the precomputed pairwise distances tensor and the trust region
        # size `eps` (or a hard-coded value of 1 in the case of categorical spaces).

        # Indices of the neighbors within the trust region.
        neighbors_mask = tf.reduce_all(
            tf.gather(global_distances, self._location_ix) <= distance, axis=-1
        )
        neighbors_mask = tf.reshape(neighbors_mask, (-1,))
        neighbor_ixs = tf.where(neighbors_mask)
        neighbor_ixs = tf.squeeze(neighbor_ixs, axis=-1)
        # Points within the trust region (including the location point).
        return tf.gather(self.global_search_space.points, neighbor_ixs)


class FixedPointTrustRegionDiscrete(UpdatableTrustRegionDiscrete):
    """
    A discrete trust region with a fixed point location that does not change across active learning
    steps. The fixed point is selected at random from the global (discrete) search space at
    initialization time.
    """

    def __init__(
        self,
        global_search_space: GeneralDiscreteSearchSpace,
        region_index: Optional[int] = None,
        input_active_dims: Optional[Union[slice, Sequence[int]]] = None,
    ):
        super().__init__(global_search_space, region_index, input_active_dims)
        # Random initial point from the global search space.
        self._init_location()

    def initialize(
        self,
        models: Optional[Mapping[Tag, ProbabilisticModelType]] = None,
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> None:
        # Pick a random point from the global search space.
        self._init_location(models, datasets)
        self._initialized = True

    def update(
        self,
        models: Optional[Mapping[Tag, ProbabilisticModelType]] = None,
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> None:
        # Keep the point fixed, no updates needed.
        pass


class SingleObjectiveTrustRegionDiscrete(UpdatableTrustRegionDiscrete, HypercubeTrustRegion):
    """
    An updatable discrete trust region that maintains a set of neighboring points around a
    single location point, allowing for local exploration of the search space. The region is
    updated based on the best point found in the region.

    This trust region is designed for discrete numerical variables. As it uses axis-aligned
    Euclidean distance to determine the neighbors within the region, it is not suitable for
    qualitative (categorical, ordinal and binary) variables.

    When using this trust region, it is important to consider the scaling of the number of value
    combinations. Since the region computes pairwise distances between points, the computational
    and memory complexity increases quadratically with the number of points. For example,
    1000 3D points will result in the distances matrix containing 1000x1000x3 entries. Therefore,
    this trust region is not suitable for problems with a large number of points.
    """

    def __init__(
        self,
        global_search_space: GeneralDiscreteSearchSpace,
        beta: float = 0.7,
        kappa: float = 1e-4,
        zeta: float = 0.5,
        min_eps: float = 1e-2,
        region_index: Optional[int] = None,
        input_active_dims: Optional[Union[slice, Sequence[int]]] = None,
    ):
        """
        Select a random initial location from the global search space and select the initial
        neighbors within the trust region.

        :param global_search_space: The global search space this search space lives in.
        :param beta: The inverse of the trust region contraction factor.
        :param kappa: Scales the threshold for the minimal improvement required for a step to be
            considered a success.
        :param zeta: The initial size of the trust region is ``zeta`` times the size of the global
            search space.
        :param min_eps: The minimal size of the search space. If the size of the search space is
            smaller than this, the search space is reinitialized.
        :param region_index: The index of the region in a multi-region search space. This is used to
            identify the local models and datasets to use for acquisition. If `None`, the
            global models and datasets are used.
        :param input_active_dims: The active dimensions of the input space, either a slice or list
            of indices into the columns of the space. If `None`, all dimensions are active.
        """
        UpdatableTrustRegionDiscrete.__init__(
            self, global_search_space, region_index, input_active_dims
        )
        # Need to compute the distances before initializing `HypercubeTrustRegion` as it
        # uses the distances to set the initial location and update the bounds.
        self._global_distances = self._compute_global_distances()
        HypercubeTrustRegion.__init__(self, beta, kappa, zeta, min_eps)

    def _update_domain(self) -> None:
        self._points = self._get_points_within_distance(self._global_distances, self.eps)


class UpdatableTrustRegionProduct(TaggedProductSearchSpace, UpdatableTrustRegion):
    """
    An updatable mixed search space that is the product of multiple updatable trust sub-regions.

    This is useful for combining different types of search spaces, such as continuous and discrete,
    to form a mixed search space for trust region acquisition rules.

    Note: the dtype of all the component search spaces must be the same.
    """

    def __init__(
        self,
        regions: Sequence[UpdatableTrustRegion],
        tags: Optional[Sequence[str]] = None,
        region_index: Optional[int] = None,
    ):
        """
        :param regions: The trust sub-regions to be combined to create a product trust region.
        :param tags: An optional list of tags giving the unique identifiers of the region's
            sub-regions.
        :param region_index: The index of the region in a multi-region search space. This is used to
            identify the local models and datasets to use for acquisition. If `None`, the
            global models and datasets are used.
        """
        assert len(regions) > 0, "at least one region should be provided"

        # If set, assert all regions have the same index and matching the product index.
        if region_index is not None:
            assert all(
                region.region_index == region_index
                for region in regions
                if region.region_index is not None
            ), (
                "all regions should have the same index, if set, as the "
                f"product region ({region_index})"
            )
        else:
            assert all(region.region_index is None for region in regions), (
                f"regions can only have a region_index if the product region ({region_index}) "
                "has one"
            )

        self._global_search_space = TaggedProductSearchSpace(
            [region.global_search_space for region in regions], tags
        )

        TaggedProductSearchSpace.__init__(self, regions)
        # When UpdatableTrustRegion sets the region_index, it will also set the region_index for
        # each region.
        # Setting of input active dims is not supported for product regions. All input dims
        # are always active.
        UpdatableTrustRegion.__init__(self, region_index, input_active_dims=None)

        # Set active dimensions for each sub-region.
        dim_ix = 0
        for region, dims in zip(regions, self.subspace_dimension):
            # Check the region's input active dims are not already set.
            assert region.input_active_dims is None, (
                f"input_active_dims ({region.input_active_dims}) should not be set for sub-regions "
                f" ({region}) of a product region"
            )
            region.input_active_dims = slice(dim_ix, dim_ix + dims)
            dim_ix += dims

    @property
    def requires_initialization(self) -> bool:
        """
        Return `True` if the search space needs to be initialized, and `False` otherwise.

        Re-initialize the whole product region if any of the sub-regions need to be re-initialized.
        """
        return any(region.requires_initialization for region in self.regions.values())

    @property
    def region_index(self) -> Optional[int]:
        """The index of the region in a multi-region search space."""
        return self._region_index

    @region_index.setter
    def region_index(self, region_index: Optional[int]) -> None:
        """Set the index of the region in a multi-region search space, including all sub-regions."""
        self._region_index = region_index
        # Override the region index for each sub-region. These would either already be set to the
        # same value (assert in __init__), or None.
        for region in self.regions.values():
            region.region_index = region_index

    @property
    def regions(self) -> Mapping[str, UpdatableTrustRegion]:
        """The sub-regions of the product trust region."""
        _regions = {}
        for tag, region in self._spaces.items():
            assert isinstance(region, (UpdatableTrustRegionBox, UpdatableTrustRegionDiscrete))
            _regions[tag] = region
        return _regions

    @property
    def location(self) -> TensorType:
        """
        The location of the product trust region, concatenated from the locations of the
        sub-regions.
        """
        return tf.concat([region.location for region in self.regions.values()], axis=-1)

    @location.setter
    def location(self, location: TensorType) -> None:
        raise NotImplementedError("setting the location of a product region is not supported")

    @property
    def global_search_space(self) -> TaggedProductSearchSpace:
        """The global search space this search space lives in."""
        return self._global_search_space

    def initialize(
        self,
        models: Optional[Mapping[Tag, ProbabilisticModelType]] = None,
        datasets: Optional[Mapping[Tag, Dataset]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        for region in self.regions.values():
            region.initialize(models, datasets, *args, **kwargs)
        # This is not used for the product region, but set it for consistency.
        self._initialized = True

    def update(
        self,
        models: Optional[Mapping[Tag, ProbabilisticModelType]] = None,
        datasets: Optional[Mapping[Tag, Dataset]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        for region in self.regions.values():
            region.update(models, datasets, *args, **kwargs)

    def get_datasets_filter_mask(
        self, datasets: Optional[Mapping[Tag, Dataset]]
    ) -> Optional[Mapping[Tag, tf.Tensor]]:
        # Return a boolean AND of the masks of each sub-region.

        # Only select the region datasets for filtering. Don't directly filter the global dataset.
        assert (
            self.region_index is not None
        ), "the region_index should be set for filtering local datasets"

        # Mask for each sub-region.
        masks = [region.get_datasets_filter_mask(datasets) for region in self.regions.values()]

        if masks[0] is not None:  # There is always at least one region.
            assert all(
                set(mask.keys()) == set(masks[0].keys()) for mask in masks if mask is not None
            ), "all region masks should have the same keys"

            return {
                tag: tf.reduce_all([mask[tag] for mask in masks if mask is not None], axis=0)
                for tag in masks[0].keys()
            }
        else:
            return None


class BatchTrustRegionProduct(
    BatchTrustRegion[ProbabilisticModelType, UpdatableTrustRegionProduct]
):
    """
    Implements the :class:`BatchTrustRegion` *trust region* acquisition rule for mixed search
    spaces. This is intended to be used for single-objective optimization with batching.
    """

    def initialize_subspaces(self, search_space: SearchSpaceType) -> None:
        if self._init_subspaces is None:
            # If no initial subspaces were provided, create N default subspaces, where N is the
            # number of query points in the base-rule.
            # Currently the detection for N is only implemented for EGO.
            # Note: the reason we don't create the default subspaces in `__init__` is because we
            # don't have the global search space at that point.
            if isinstance(self._rule, EfficientGlobalOptimization):
                num_query_points = self._rule._num_query_points
            else:
                num_query_points = 1

            def create_subregions() -> Sequence[UpdatableTrustRegion]:
                # Take a global product search space and convert each of its subspaces to an
                # updatable trust sub-region. These sub-regions are then used to create a
                # trust region product.
                assert isinstance(
                    search_space, TaggedProductSearchSpace
                ), "search_space should be a TaggedProductSearchSpace"

                subregions: List[UpdatableTrustRegion] = []
                for tag in search_space.subspace_tags:
                    subspace = search_space.get_subspace(tag)
                    if isinstance(subspace, DiscreteSearchSpace):
                        subregions.append(FixedPointTrustRegionDiscrete(subspace))
                    elif isinstance(subspace, Box):
                        subregions.append(SingleObjectiveTrustRegionBox(subspace))
                    else:
                        raise ValueError(f"unsupported search space type: {type(subspace)}")
                return subregions

            init_subspaces: Tuple[UpdatableTrustRegionProduct, ...] = tuple(
                UpdatableTrustRegionProduct(create_subregions()) for _ in range(num_query_points)
            )
            self._init_subspaces = init_subspaces
            for index, subspace in enumerate(self._init_subspaces):
                subspace.region_index = index  # Override the index.
            self._tags = tuple(str(index) for index in range(self.num_local_datasets))

    def acquire(
        self,
        search_space: SearchSpace,
        models: Mapping[Tag, ProbabilisticModelType],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> types.State[BatchTrustRegionState[UpdatableTrustRegionProduct] | None, TensorType]:
        # Ensure passed in global search space is always the same as the search space passed to
        # the subspaces.
        if self._init_subspaces is not None:
            for subspace in self._init_subspaces:
                assert subspace.global_search_space == search_space, (
                    "The global search space of the subspaces should be the same as the "
                    "search space passed to the BatchTrustRegionProduct acquisition rule. "
                    "If you want to change the global search space, you should recreate the rule. "
                    "Note: all subspaces should be initialized with the same global search space."
                )

        return super().acquire(search_space, models, datasets)

    @inherit_check_shapes
    def get_initialize_subspaces_mask(
        self,
        subspaces: Sequence[UpdatableTrustRegionProduct],
        models: Mapping[Tag, ProbabilisticModelType],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> TensorType:
        # Initialize the subspaces that have non-unique locations.
        centres = tf.stack([subspace.location for subspace in subspaces])
        return tf.logical_not(get_unique_points_mask(centres, tolerance=1e-6))


class BatchHypervolumeSharpeRatioIndicator(
    AcquisitionRule[TensorType, SearchSpace, ProbabilisticModel]
):
    """Implements the Batch Hypervolume Sharpe-ratio indicator acquisition
    rule, designed for large batches, introduced by Binois et al, 2021.
    See :cite:`binois2021portfolio` for details.
    """

    def __init__(
        self,
        num_query_points: int = 1,
        ga_population_size: int = 500,
        ga_n_generations: int = 200,
        filter_threshold: float = 0.1,
        noisy_observations: bool = True,
    ):
        """
        :param num_query_points: The number of points in a batch. Defaults to 5.
        :param ga_population_size: The population size used in the genetic algorithm
             that finds points on the Pareto front. Defaults to 500.
        :param ga_n_generations: The number of genenrations to run in the genetic
             algorithm. Defaults to 200.
        :param filter_threshold: The probability of improvement below which to exlude
             points from the Sharpe ratio optimisation. Defaults to 0.1.
        :param noisy_observations: Whether the observations have noise. Defaults to True.
        """
        if num_query_points <= 0:
            raise ValueError(f"Num query points must be greater than 0, got {num_query_points}")
        if ga_population_size < num_query_points:
            raise ValueError(
                "Population size must be greater or equal to num query points size, got num"
                f" query points as {num_query_points} and population size as {ga_population_size}"
            )
        if ga_n_generations <= 0:
            raise ValueError(f"Number of generation must be greater than 0, got {ga_n_generations}")
        if not 0.0 <= filter_threshold < 1.0:
            raise ValueError(f"Filter threshold must be in range [0.0,1.0), got {filter_threshold}")
        if pymoo is None:
            raise ImportError(
                "BatchHypervolumeSharpeRatioIndicator requires pymoo, "
                "which can be installed via `pip install trieste[qhsri]`"
            )
        builder = ProbabilityOfImprovement().using(OBJECTIVE)

        self._builder: AcquisitionFunctionBuilder[ProbabilisticModel] = builder
        self._num_query_points: int = num_query_points
        self._population_size: int = ga_population_size
        self._n_generations: int = ga_n_generations
        self._filter_threshold: float = filter_threshold
        self._noisy_observations: bool = noisy_observations
        self._acquisition_function: Optional[AcquisitionFunction] = None

    def __repr__(self) -> str:
        """"""
        return f"""BatchHypervolumeSharpeRatioIndicator(
        num_query_points={self._num_query_points}, ga_population_size={self._population_size},
        ga_n_generations={self._n_generations}, filter_threshold={self._filter_threshold},
        noisy_observations={self._noisy_observations}
        )
        """

    def _find_non_dominated_points(
        self, model: ProbabilisticModel, search_space: SearchSpaceType
    ) -> tuple[TensorType, TensorType]:
        """Uses NSGA-II to find high-quality non-dominated points"""

        problem = _MeanStdTradeoff(model, search_space)
        algorithm = NSGA2(pop_size=self._population_size)
        res = minimize(problem, algorithm, ("n_gen", self._n_generations), seed=1, verbose=False)

        return res.X, res.F

    def _filter_points(
        self, nd_points: TensorType, nd_mean_std: TensorType
    ) -> tuple[TensorType, TensorType]:
        if self._acquisition_function is None:
            raise ValueError("Acquisition function has not been defined yet")

        probs_of_improvement = np.array(
            self._acquisition_function(np.expand_dims(nd_points, axis=-2))
        )

        above_threshold = probs_of_improvement > self._filter_threshold

        if np.sum(above_threshold) >= self._num_query_points and nd_mean_std.shape[1] == 2:
            # There are enough points above the threshold to get a batch
            out_points, out_mean_std = (
                nd_points[above_threshold.squeeze(), :],
                nd_mean_std[above_threshold.squeeze(), :],
            )
        else:
            # We don't filter
            out_points, out_mean_std = nd_points, nd_mean_std

        return out_points, out_mean_std

    def acquire(
        self,
        search_space: SearchSpace,
        models: Mapping[Tag, ProbabilisticModel],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> TensorType:
        """Acquire a batch of points to observe based on the batch hypervolume
        Sharpe ratio indicator method.
        This method uses NSGA-II to create a Pareto set of the mean and standard
        deviation of the posterior of the probabilistic model, and then selects
        points to observe based on maximising the Sharpe ratio.

        :param search_space: The local acquisition search space for *this step*.
        :param models: The model for each tag.
        :param datasets: The known observer query points and observations.
        :return: The batch of points to query.
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
                models, datasets=datasets
            )
        else:
            self._acquisition_function = self._builder.update_acquisition_function(
                self._acquisition_function,
                models,
                datasets=datasets,
            )

        # Find non-dominated points
        nd_points, nd_mean_std = self._find_non_dominated_points(models[OBJECTIVE], search_space)

        # Filter out points below a threshold probability of improvement
        filtered_points, filtered_mean_std = self._filter_points(nd_points, nd_mean_std)

        # Set up a Pareto set of the filtered points
        pareto_set = Pareto(filtered_mean_std, already_non_dominated=True)

        # Sample points from set using qHSRI
        _, batch_ids = pareto_set.sample_diverse_subset(
            self._num_query_points, allow_repeats=self._noisy_observations
        )

        batch = filtered_points[batch_ids]

        return batch


class _MeanStdTradeoff(PymooProblem):  # type: ignore[misc]
    """Inner class that formulates the mean/std optimisation problem as a
    pymoo problem"""

    def __init__(self, probabilistic_model: ProbabilisticModel, search_space: SearchSpaceType):
        """
        :param probabilistic_model: The probabilistic model to find optimal mean/stds from
        :param search_space: The search space for the optimisation
        """
        # If we have a stack of models we have mean and std for each
        if isinstance(probabilistic_model, ModelStack):
            n_obj = 2 * len(probabilistic_model._models)
        else:
            n_obj = 2
        super().__init__(
            n_var=int(search_space.dimension),
            n_obj=n_obj,
            n_constr=0,
            xl=np.array(search_space.lower),
            xu=np.array(search_space.upper),
        )
        self.probabilistic_model = probabilistic_model

    def _evaluate(
        self, x: TensorType, out: dict[str, TensorType], *args: Any, **kwargs: Any
    ) -> None:
        mean, var = self.probabilistic_model.predict(x)
        # Flip sign on std so that minimising is increasing std
        std = -1 * np.sqrt(np.array(var))
        out["F"] = np.concatenate([np.array(mean), std], axis=1)
