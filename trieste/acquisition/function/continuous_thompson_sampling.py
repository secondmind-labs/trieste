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
This module contains local penalization-based acquisition function builders. TODO
"""
from __future__ import annotations

from typing import Optional

import tensorflow as tf

from ...data import Dataset
from ...models import ProbabilisticModel
from ...types import TensorType
from ..interface import AcquisitionFunction, SingleModelGreedyAcquisitionBuilder


class GreedyContinuousThompsonSampling(SingleModelGreedyAcquisitionBuilder[ProbabilisticModel]):
    r"""
    SAY IGNORE PENDING:)
    """

    def prepare_acquisition_function(
        self,
        model: ProbabilisticModel,
        dataset: Optional[Dataset] = None,
        pending_points: Optional[TensorType] = None,
    ) -> AcquisitionFunction:
        """
        :param model: The model.
        :param dataset: The data from the observer (not used).
        :param pending_points: The points already in the current batch (not used).
        :return: The (log) expected improvement penalized with respect to the pending points. TODO
        """

        try:
            self._trajectory_sampler = model.trajectory_sampler()
            trajectory = self._trajectory_sampler.get_negative_trajectory()
        except (NotImplementedError):
            raise ValueError(
                """
            Thompson sampling from trajectory only supports models with a
            trajectory_sampler method.
            """
            )

        return trajectory

    def update_acquisition_function(
        self,
        function: AcquisitionFunction,
        model: ProbabilisticModel,
        dataset: Optional[Dataset] = None,
        pending_points: Optional[TensorType] = None,
        new_optimization_step: bool = True,
    ) -> AcquisitionFunction:
        """
        :param function: The acquisition function to update.
        :param model: The model.
        :param dataset: The data from the observer. Must not be populated.
        :param pending_points: The points already in the current batch (not used).
        :param new_optimization_step: Indicates whether this call to update_acquisition_function
            is to start of a new optimization step, of to continue collecting batch of points
            for the current step. Defaults to ``True``.
        :return: The updated acquisition function.
        """
        tf.debugging.Assert(self._trajectory_sampler is not None, [])

        trajectory = function
        if new_optimization_step:
            trajectory = self._trajectory_sampler.update_trajectory(trajectory)
        else:
            trajectory = self._trajectory_sampler.resample_trajectory(trajectory)

        return trajectory
