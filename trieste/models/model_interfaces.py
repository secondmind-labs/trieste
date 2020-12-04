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
from abc import abstractmethod, ABC
from typing import Callable, Dict, Iterable, Optional, Tuple, Union, Any

import gpflow
from dataclasses import dataclass
from gpflow.models import GPModel, GPR, SGPR, VGP, SVGP
import tensorflow as tf

from .optimizer import Optimizer
from .. import utils
from ..data import Dataset
from ..type import ObserverEvaluations, QueryPoints, TensorType


class ProbabilisticModel(tf.Module, ABC):
    """ A probabilistic model. """

    @abstractmethod
    def optimize(self, dataset: Dataset) -> None:
        """
        Optimize the model.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, dataset: Dataset) -> None:
        """
        Update the model given the specified ``dataset``. Does not train the model.

        :param dataset: The data with which to update the model.
        """
        raise NotImplementedError

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


class GPflowPredictor(ProbabilisticModel, ABC):
    """ A trainable wrapper for a GPflow Gaussian process model. """

    def __init__(self, optimizer: Optional[Optimizer] = None):
        if optimizer is None:
            optimizer = Optimizer(gpflow.optimizers.Scipy())
        self._optimizer = optimizer

    @property
    def optimizer(self) -> Optimizer:
        return self._optimizer

    @property
    @abstractmethod
    def model(self) -> GPModel:
        """ The underlying GPflow model. """

    def predict(self, query_points: QueryPoints) -> Tuple[ObserverEvaluations, TensorType]:
        return self.model.predict_f(query_points)

    def sample(self, query_points: QueryPoints, num_samples: int) -> ObserverEvaluations:
        return self.model.predict_f_samples(query_points, num_samples)

    def optimize(self, dataset: Dataset):
        self.optimizer.optimize(self.model, dataset)


class GaussianProcessRegression(GPflowPredictor):
    def __init__(self, model: Union[GPR, SGPR], optimizer: Optional[Optimizer] = None):
        """
        :param model: The GPflow model to wrap.
        """
        super().__init__(optimizer)
        self._model = model

    @property
    def model(self) -> Union[GPR, SGPR]:
        return self._model

    def update(self, dataset: Dataset) -> None:
        x, y = self.model.data

        if dataset.query_points.shape[-1] != x.shape[-1]:
            raise ValueError

        if dataset.observations.shape[-1] != y.shape[-1]:
            raise ValueError

        self.model.data = dataset.query_points, dataset.observations


class SparseVariational(GPflowPredictor):
    def __init__(self, model: SVGP, data: Dataset, optimizer: Optional[Optimizer] = None):
        """
        :param optimizer: The optimizer to use for optimization.
        :param model: The underlying GPflow sparse variational model.
        :param data: The initial training data.
        :param iterations: The number of iterations for which to optimize the model.
        :param batcher: A function to convert training data into (mini)batches for optimization.
        """
        super().__init__(optimizer)
        self._model = model
        self._data = data

    @property
    def model(self) -> SVGP:
        return self._model

    def update(self, dataset: Dataset) -> None:
        if dataset.query_points.shape[-1] != self._data.query_points.shape[-1]:
            raise ValueError

        if dataset.observations.shape[-1] != self._data.observations.shape[-1]:
            raise ValueError

        self._data = dataset

        num_data = dataset.query_points.shape[0]
        self.model.num_data = num_data


class VariationalGaussianProcess(GaussianProcessRegression):
    def update(self, dataset: Dataset):
        model = self.model
        x, y = model.data
        assert dataset.query_points.shape[-1] == x.shape[-1]
        assert dataset.observations.shape[-1] == y.shape[-1]
        data = (dataset.query_points, dataset.observations)
        num_data = data[0].shape[0]

        f_mu, f_cov = self.model.predict_f(dataset.query_points, full_cov=True)  # [N, L], [L, N, N]
        assert self.model.q_sqrt.shape.ndims == 3

        # GPflow's VGP model is hard-coded to use the whitened representation, i.e.
        # q_mu and q_sqrt parametrise q(v), and u = f(X) = L v, where L = cholesky(K(X, X))
        # Hence we need to backtransform from f_mu and f_cov to obtain the updated
        # new_q_mu and new_q_sqrt:
        Knn = model.kernel(dataset.query_points, full_cov=True)  # [N, N]
        jitter_mat = gpflow.config.default_jitter() * tf.eye(num_data, dtype=Knn.dtype)
        Lnn = tf.linalg.cholesky(Knn + jitter_mat)  # [N, N]
        new_q_mu = tf.linalg.triangular_solve(Lnn, f_mu)  # [N, L]
        tmp = tf.linalg.triangular_solve(Lnn[None], f_cov)  # [L, N, N], L⁻¹ f_cov
        S_v = tf.linalg.triangular_solve(Lnn[None], tf.linalg.matrix_transpose(tmp))  # [L, N, N]
        new_q_sqrt = tf.linalg.cholesky(S_v + jitter_mat)  # [L, N, N]

        model.data = data
        model.num_data = num_data
        model.q_mu = gpflow.Parameter(new_q_mu)
        model.q_sqrt = gpflow.Parameter(new_q_sqrt, transform=gpflow.utilities.triangular())

    def predict(self, query_points: QueryPoints) -> Tuple[ObserverEvaluations, TensorType]:
        return self.model.predict_y(query_points)


supported_models: Dict[Any, Callable[[Any], ProbabilisticModel]] = {
    GPR: GaussianProcessRegression,
    SGPR: GaussianProcessRegression,
    VGP: VariationalGaussianProcess,
}
"""
:var supported_models: A mapping of third-party model types to :class:`CustomTrainable` classes
that wrap models of those types.
"""
