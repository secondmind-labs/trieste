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
This module contains acquisition function builders for continuous Thompson sampling.
"""
from __future__ import annotations

from typing import Optional

from ...data import Dataset
from ...models.interfaces import HasTrajectorySampler, TrajectoryFunction, TrajectoryFunctionClass
from ...types import TensorType
from ..interface import SingleModelGreedyAcquisitionBuilder, SingleModelVectorizedAcquisitionBuilder


class GreedyContinuousThompsonSampling(SingleModelGreedyAcquisitionBuilder[HasTrajectorySampler]):
    r"""

    Acquisition function builder for performing greedy continuous Thompson sampling. This builder
    return acquisition functions that are the negatives of approximate samples from the
    given :class:`ProbabilisticModel`, as provided by the model's :meth:`get_trajectory`
    method. A set of such samples are to be maximized in a sequential greedy manner to provide
    the next recommended query points. Note that we actually return
    the negative of the trajectory, so that our acquisition optimizers (which are
    all maximizers) can be used to extract the minimisers of trajectories.


    For more details about trajectory-based Thompson sampling see :cite:`hernandez2017parallel` and
    :cite:`wilson2020efficiently`.
    """

    def prepare_acquisition_function(
        self,
        model: HasTrajectorySampler,
        dataset: Optional[Dataset] = None,
        pending_points: Optional[TensorType] = None,
    ) -> TrajectoryFunction:
        """
        :param model: The model.
        :param dataset: The data from the observer (not used).
        :param pending_points: The points already in the current batch (not used).
        :return: A negated trajectory sampled from the model.
        """
        if not isinstance(model, HasTrajectorySampler):
            raise ValueError(
                f"Thompson sampling from trajectory only supports models with a trajectory_sampler "
                f"method; received {model.__repr__()}"
            )

        self._trajectory_sampler = model.trajectory_sampler()
        function = self._trajectory_sampler.get_trajectory()
        return negate_trajectory_function(function)

    def update_acquisition_function(
        self,
        function: TrajectoryFunction,
        model: HasTrajectorySampler,
        dataset: Optional[Dataset] = None,
        pending_points: Optional[TensorType] = None,
        new_optimization_step: bool = True,
    ) -> TrajectoryFunction:
        """
        :param function: The trajectory function to update.
        :param model: The model.
        :param dataset: The data from the observer (not used).
        :param pending_points: The points already in the current batch (not used).
        :param new_optimization_step: Indicates whether this call to update_acquisition_function
            is to start of a new optimization step, of to continue collecting batch of points
            for the current step. Defaults to ``True``.
        :return: A new trajectory sampled from the model.
        """

        if new_optimization_step:  # update sampler and resample trajectory
            new_function = self._trajectory_sampler.update_trajectory(function)
        else:  # just resample trajectory but without updating sampler
            new_function = self._trajectory_sampler.resample_trajectory(function)

        if new_function is not function:
            function = negate_trajectory_function(new_function)

        return function


class ParallelContinuousThompsonSampling(
    SingleModelVectorizedAcquisitionBuilder[HasTrajectorySampler]
):
    r"""
    Acquisition function builder for performing parallel continuous Thompson sampling.

    This builder provides broadly the same behavior as our :class:`GreedyContinuousThompsonSampler`
    however optimizes trajectory samples in parallel rather than sequentially.
    Consequently, :class:`ParallelContinuousThompsonSampling` can choose query points faster
    than  :class:`GreedyContinuousThompsonSampler` however it has much larger memory usage.

    For a convenient way to control the total memory usage of this acquisition function, see
    our :const:`split_acquisition_function_calls` wrapper.
    """

    def prepare_acquisition_function(
        self,
        model: HasTrajectorySampler,
        dataset: Optional[Dataset] = None,
    ) -> TrajectoryFunction:
        """
        :param model: The model.
        :param dataset: The data from the observer (not used).
        :return: A negated trajectory sampled from the model.
        """
        if not isinstance(model, HasTrajectorySampler):
            raise ValueError(
                f"Thompson sampling from trajectory only supports models with a trajectory_sampler "
                f"method; received {model.__repr__()}"
            )

        self._trajectory_sampler = model.trajectory_sampler()
        function = self._trajectory_sampler.get_trajectory()
        return negate_trajectory_function(function)

    def update_acquisition_function(
        self,
        function: TrajectoryFunction,
        model: HasTrajectorySampler,
        dataset: Optional[Dataset] = None,
    ) -> TrajectoryFunction:
        """
        :param function: The trajectory function to update.
        :param model: The model.
        :param dataset: The data from the observer (not used).
        :return: A new trajectory sampled from the model.
        """

        new_function = self._trajectory_sampler.update_trajectory(function)

        if new_function is not function:
            function = negate_trajectory_function(new_function)

        return function


def negate_trajectory_function(function: TrajectoryFunction) -> TrajectoryFunction:
    """
    Return the negative of trajectories so that our acquisition optimizers (which are
    all maximizers) can be used to extract the minimizers of trajectories.

    We negate the trajectory function object's call method but otherwise leave it alone,
    as it may have e.g. update and resample methods.
    """
    if isinstance(function, TrajectoryFunctionClass):

        class NegatedTrajectory(type(function)):  # type: ignore[misc]
            def __call__(self, x: TensorType) -> TensorType:
                return -1.0 * super().__call__(x)

        function.__class__ = NegatedTrajectory

    else:

        def negated_trajectory(x: TensorType) -> TensorType:
            return -1.0 * function(x)

        function = negated_trajectory

    return function
