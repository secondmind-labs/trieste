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

from ...types import TensorType
from ..interfaces import EnsembleModel, TrajectoryFunction, TrajectorySampler


class EnsembleTrajectorySampler(TrajectorySampler[EnsembleModel]):
    """
    This class builds functions that approximate a trajectory by randomly choosing a network from
    the ensemble and using its predicted means as a trajectory.
    """

    def __init__(self, model: EnsembleModel):
        """
        :param model: The ensemble model to sample from.
        """
        if not isinstance(model, EnsembleModel):
            raise NotImplementedError(
                f"EnsembleTrajectorySampler only works with EnsembleModel models, that support "
                f"ensemble_size, sample_index, predict_ensemble and sample_ensemble methods; "
                f"received {model.__repr__()}"
            )

        super().__init__(model)

        self._model = model

    def __repr__(self) -> str:
        """"""
        return f"{self.__class__.__name__}({self._model!r}"

    def get_trajectory(self) -> TrajectoryFunction:
        """
        Generate an approximate function draw (trajectory) by randomly choosing a network from
        the ensemble and using its predicted means as a trajectory.

        :return: A trajectory function representing an approximate trajectory from the ensemble
            model, taking an input of shape `[N, D]` and returning shape `[N, 1]`
        """
        network_index = self._model.sample_index(1)[0]

        def trajectory(x: TensorType) -> TensorType:
            return self._model.predict_ensemble(x)[network_index][0]

        return trajectory
