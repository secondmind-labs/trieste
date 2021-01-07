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
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple, Union

import gpflow
import tensorflow as tf
from gpflow.models import GPR, SGPR, SVGP, VGP, GPModel

from ..data import Dataset
from ..type import TensorType
from .optimizer import Optimizer


class ProbabilisticModel(tf.Module, ABC):
    """ A probabilistic model. """

    @abstractmethod
    def predict(self, query_points: TensorType) -> Tuple[TensorType, TensorType]:
        """
        Return the predicted mean and variance of the latent function(s) at the specified
        ``query_points``.

        :param query_points: The points at which to make predictions.
        :return: The predicted mean and variance.
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, query_points: TensorType, num_samples: int) -> TensorType:
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
    def optimize(self, dataset: Dataset) -> None:
        """
        Optimize the model objective with respect to (hyper)parameters given the specified
        ``dataset``.

        :param dataset: The data with which to train the model.
        """
        raise NotImplementedError


class GPflowPredictor(ProbabilisticModel, ABC):
    """ A trainable wrapper for a GPflow Gaussian process model. """

    def __init__(self, optimizer: Optional[Optimizer] = None):
        """
        :param optimizer: The optimizer with which to train the model. Defaults to
            :class:`~trieste.models.optimizer.Optimizer` with :class:`~gpflow.optimizers.Scipy`.
        """
        super().__init__()

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

    def predict(self, query_points: TensorType) -> Tuple[TensorType, TensorType]:
        return self.model.predict_f(query_points)

    def sample(self, query_points: TensorType, num_samples: int) -> TensorType:
        return self.model.predict_f_samples(query_points, num_samples)

    def optimize(self, dataset: Dataset) -> None:
        self.optimizer.optimize(self.model, dataset)


class GaussianProcessRegression(GPflowPredictor, TrainableProbabilisticModel):
    def __init__(self, model: Union[GPR, SGPR], optimizer: Optional[Optimizer] = None):
        """
        :param model: The GPflow model to wrap.
        :param optimizer: The optimizer with which to train the model. Defaults to
            :class:`~trieste.models.optimizer.Optimizer` with :class:`~gpflow.optimizers.Scipy`.
        """
        super().__init__(optimizer)
        self._model = model

    def __repr__(self) -> str:
        return f"GaussianProcessRegression({self._model!r}, {self.optimizer!r})"

    @property
    def model(self) -> Union[GPR, SGPR]:
        return self._model

    def update(self, dataset: Dataset) -> None:
        x, y = self.model.data

        _assert_data_is_compatible(dataset, Dataset(x, y))

        if dataset.query_points.shape[-1] != x.shape[-1]:
            raise ValueError

        if dataset.observations.shape[-1] != y.shape[-1]:
            raise ValueError

        self.model.data = dataset.query_points, dataset.observations


class SparseVariational(GPflowPredictor, TrainableProbabilisticModel):
    def __init__(self, model: SVGP, data: Dataset, optimizer: Optional[Optimizer] = None):
        """
        :param optimizer: The optimizer with which to train the model. Defaults to
            :class:`~trieste.models.optimizer.Optimizer` with :class:`~gpflow.optimizers.Scipy`.
        :param model: The underlying GPflow sparse variational model.
        :param data: The initial training data.
        """
        super().__init__(optimizer)
        self._model = model
        self._data = data

    def __repr__(self) -> str:
        return f"SparseVariational({self._model!r}, {self._data!r}, {self.optimizer!r})"

    @property
    def model(self) -> SVGP:
        return self._model

    def update(self, dataset: Dataset) -> None:
        _assert_data_is_compatible(dataset, self._data)

        self._data = dataset

        num_data = dataset.query_points.shape[0]
        self.model.num_data = num_data


class VariationalGaussianProcess(GaussianProcessRegression):
    def __init__(self, model: VGP, optimizer: Optional[Optimizer] = None):
        tf.debugging.assert_rank(self.model.q_sqrt, 3)
        super().__init__(model, optimizer)

    def __repr__(self) -> str:
        return f"VariationalGaussianProcess({self._model!r}, {self.optimizer!r})"

    def update(self, dataset: Dataset) -> None:
        model = self.model

        _assert_data_is_compatible(dataset, Dataset(*model.data))

        f_mu, f_cov = self.model.predict_f(dataset.query_points, full_cov=True)  # [N, L], [L, N, N]

        # GPflow's VGP model is hard-coded to use the whitened representation, i.e.
        # q_mu and q_sqrt parametrise q(v), and u = f(X) = L v, where L = cholesky(K(X, X))
        # Hence we need to backtransform from f_mu and f_cov to obtain the updated
        # new_q_mu and new_q_sqrt:
        Knn = model.kernel(dataset.query_points, full_cov=True)  # [N, N]
        jitter_mat = gpflow.config.default_jitter() * tf.eye(len(dataset), dtype=Knn.dtype)
        Lnn = tf.linalg.cholesky(Knn + jitter_mat)  # [N, N]
        new_q_mu = tf.linalg.triangular_solve(Lnn, f_mu)  # [N, L]
        tmp = tf.linalg.triangular_solve(Lnn[None], f_cov)  # [L, N, N], L⁻¹ f_cov
        S_v = tf.linalg.triangular_solve(Lnn[None], tf.linalg.matrix_transpose(tmp))  # [L, N, N]
        new_q_sqrt = tf.linalg.cholesky(S_v + jitter_mat)  # [L, N, N]

        model.data = dataset.astuple()
        model.num_data = len(dataset)
        model.q_mu = gpflow.Parameter(new_q_mu)
        model.q_sqrt = gpflow.Parameter(new_q_sqrt, transform=gpflow.utilities.triangular())

    def predict(self, query_points: TensorType) -> Tuple[TensorType, TensorType]:
        return self.model.predict_y(query_points)


supported_models: Dict[Any, Callable[[Any, Optimizer], TrainableProbabilisticModel]] = {
    GPR: GaussianProcessRegression,
    SGPR: GaussianProcessRegression,
    VGP: VariationalGaussianProcess,
}
"""
A mapping of third-party model types to :class:`CustomTrainable` classes that wrap models of those
types.
"""


def _assert_data_is_compatible(new_data: Dataset, existing_data: Dataset) -> None:
    if new_data.query_points.shape[-1] != existing_data.query_points.shape[-1]:
        raise ValueError(
            f"Shape {new_data.query_points.shape} of new query points is incompatible with"
            f" shape {existing_data.query_points.shape} of existing query points. Trailing"
            f" dimensions must match."
        )

    if new_data.observations.shape[-1] != existing_data.observations.shape[-1]:
        raise ValueError(
            f"Shape {new_data.observations.shape} of new observations is incompatible with"
            f" shape {existing_data.observations.shape} of existing observations. Trailing"
            f" dimensions must match."
        )
