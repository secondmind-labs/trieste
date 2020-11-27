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
from __future__ import annotations
from abc import abstractmethod, ABC
from typing import Tuple

from ..data import Dataset
from ..type import ObserverEvaluations, QueryPoints, TensorType


class ProbabilisticModel(ABC):
    """ A probabilistic model. """

    @abstractmethod
    def predict(self, query_points: QueryPoints) -> Tuple[ObserverEvaluations, TensorType]:
        """
        Return the predicted mean and variance of the latent function(s) at the specified
        ``query_points``.

        :param query_points: The points at which to make predictions.
        :return: The predicted mean and variance.
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, query_points: QueryPoints, num_samples: int) -> ObserverEvaluations:
        """
        Return ``num_samples`` samples from the predictive distribution at ``query_points``.

        :param query_points: The points at which to sample.
        :param num_samples: The number of samples at each point.
        :return: The samples. Has shape [S, Q, D], where S is the number of samples, Q is the number
            of query points, and D is the dimension of the predictive distribution.
        """
        raise NotImplementedError


class TrainableProbabilisticModel(ProbabilisticModel):
    """ A trainable probabilistic model. """

    @abstractmethod
    def update(self, dataset: Dataset) -> None:
        """
        Update the model given the specified ``dataset``. Does not train the model.

        :param dataset: The data with which to update the model.
        """
        raise NotImplementedError

    @abstractmethod
    def optimize(self) -> None:
        """ Optimize the model parameters. """
        raise NotImplementedError
