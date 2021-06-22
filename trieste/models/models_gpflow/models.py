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

import gpflow
import tensorflow as tf
from gpflow.models import GPR, SGPR, SVGP, VGP

from ...data import Dataset
from ...type import TensorType
from ...utils import DEFAULTS
from ..optimizer import Optimizer
from ..interfaces import TrainableProbabilisticModel
from .utils import assert_data_is_compatible
from .interface import GPflowPredictor


class GaussianProcessRegression(GPflowPredictor, TrainableProbabilisticModel):
    """
    A :class:`TrainableProbabilisticModel` wrapper for a GPflow :class:`~gpflow.models.GPR`
    or :class:`~gpflow.models.SGPR`.
    """

    def __init__(self, model: GPR | SGPR, optimizer: Optimizer | None = None):
        """
        :param model: The GPflow model to wrap.
        :param optimizer: The optimizer with which to train the model. Defaults to
            :class:`~trieste.models.optimizer.Optimizer` with :class:`~gpflow.optimizers.Scipy`.
        """
        super().__init__(optimizer)
        self._model = model

    def __repr__(self) -> str:
        """"""
        return f"GaussianProcessRegression({self._model!r}, {self.optimizer!r})"

    @property
    def model(self) -> GPR | SGPR:
        return self._model

    def update(self, dataset: Dataset) -> None:
        x, y = self.model.data

        assert_data_is_compatible(dataset, Dataset(x, y))

        if dataset.query_points.shape[-1] != x.shape[-1]:
            raise ValueError

        if dataset.observations.shape[-1] != y.shape[-1]:
            raise ValueError

        self.model.data = dataset.query_points, dataset.observations

    def covariance_between_points(
        self, query_points_1: TensorType, query_points_2: TensorType
    ) -> TensorType:
        r"""
        Compute the posterior covariance between sets of query points.

        .. math:: \Sigma_{12} = K_{12} - K_{x1}(K_{xx} + \sigma^2 I)^{-1}K_{x2}

        :param query_points_1: Set of query points with shape [N, D]
        :param query_points_2: Sets of query points with shape [M, D]

        :return: Covariance matrix between the sets of query points with shape [N, M]
        """
        tf.debugging.assert_shapes([(query_points_1, ["N", "D"]), (query_points_2, ["M", "D"])])

        x, _ = self.model.data
        num_data = x.shape[0]
        s = tf.linalg.diag(tf.fill([num_data], self.model.likelihood.variance))

        K = self.model.kernel(x)
        L = tf.linalg.cholesky(K + s)

        Kx1 = self.model.kernel(x, query_points_1)
        Linv_Kx1 = tf.linalg.triangular_solve(L, Kx1)

        Kx2 = self.model.kernel(x, query_points_2)
        Linv_Kx2 = tf.linalg.triangular_solve(L, Kx2)

        K12 = self.model.kernel(query_points_1, query_points_2)
        cov = K12 - tf.tensordot(tf.transpose(Linv_Kx1), Linv_Kx2, [[-1], [-2]])

        tf.debugging.assert_shapes(
            [(query_points_1, ["N", "D"]), (query_points_2, ["M", "D"]), (cov, ["N", "M"])]
        )

        return cov

    def get_observation_noise(self):
        """
        Return the variance of observation noise for homoscedastic likelihoods.
        :return: The observation noise.
        :raise NotImplementedError: If the model does not have a homoscedastic likelihood.
        """
        try:
            noise_variance = self.model.likelihood.variance
        except AttributeError:
            raise NotImplementedError("Model {self!r} does not have scalar observation noise")

        return noise_variance


class SparseVariational(GPflowPredictor, TrainableProbabilisticModel):
    """
    A :class:`TrainableProbabilisticModel` wrapper for a GPflow :class:`~gpflow.models.SVGP`.
    """

    def __init__(self, model: SVGP, data: Dataset, optimizer: Optimizer | None = None):
        """
        :param model: The underlying GPflow sparse variational model.
        :param data: The initial training data.
        :param optimizer: The optimizer with which to train the model. Defaults to
            :class:`~trieste.models.optimizer.Optimizer` with :class:`~gpflow.optimizers.Scipy`.
        """
        super().__init__(optimizer)
        self._model = model
        self._data = data

    def __repr__(self) -> str:
        """"""
        return f"SparseVariational({self._model!r}, {self._data!r}, {self.optimizer!r})"

    @property
    def model(self) -> SVGP:
        return self._model

    def update(self, dataset: Dataset) -> None:
        assert_data_is_compatible(dataset, self._data)

        self._data = dataset

        num_data = dataset.query_points.shape[0]
        self.model.num_data = num_data


class VariationalGaussianProcess(GPflowPredictor, TrainableProbabilisticModel):
    """A :class:`TrainableProbabilisticModel` wrapper for a GPflow :class:`~gpflow.models.VGP`."""

    def __init__(self, model: VGP, optimizer: Optimizer | None = None):
        """
        :param model: The GPflow :class:`~gpflow.models.VGP`.
        :param optimizer: The optimizer with which to train the model. Defaults to
            :class:`~trieste.models.optimizer.Optimizer` with :class:`~gpflow.optimizers.Scipy`.
        :raise ValueError (or InvalidArgumentError): If ``model``'s :attr:`q_sqrt` is not rank 3.
        """
        tf.debugging.assert_rank(model.q_sqrt, 3)
        super().__init__(optimizer)
        self._model = model

    def __repr__(self) -> str:
        """"""
        return f"VariationalGaussianProcess({self._model!r}, {self.optimizer!r})"

    @property
    def model(self) -> VGP:
        return self._model

    def update(self, dataset: Dataset, *, jitter: float = DEFAULTS.JITTER) -> None:
        """
        Update the model given the specified ``dataset``. Does not train the model.

        :param dataset: The data with which to update the model.
        :param jitter: The size of the jitter to use when stabilising the Cholesky decomposition of
            the covariance matrix.
        """
        model = self.model

        assert_data_is_compatible(dataset, Dataset(*model.data))

        f_mu, f_cov = self.model.predict_f(dataset.query_points, full_cov=True)  # [N, L], [L, N, N]

        # GPflow's VGP model is hard-coded to use the whitened representation, i.e.
        # q_mu and q_sqrt parametrise q(v), and u = f(X) = L v, where L = cholesky(K(X, X))
        # Hence we need to back-transform from f_mu and f_cov to obtain the updated
        # new_q_mu and new_q_sqrt:
        Knn = model.kernel(dataset.query_points, full_cov=True)  # [N, N]
        jitter_mat = jitter * tf.eye(len(dataset), dtype=Knn.dtype)
        Lnn = tf.linalg.cholesky(Knn + jitter_mat)  # [N, N]
        new_q_mu = tf.linalg.triangular_solve(Lnn, f_mu)  # [N, L]
        tmp = tf.linalg.triangular_solve(Lnn[None], f_cov)  # [L, N, N], L⁻¹ f_cov
        S_v = tf.linalg.triangular_solve(Lnn[None], tf.linalg.matrix_transpose(tmp))  # [L, N, N]
        new_q_sqrt = tf.linalg.cholesky(S_v + jitter_mat)  # [L, N, N]

        model.data = dataset.astuple()
        model.num_data = len(dataset)
        model.q_mu = gpflow.Parameter(new_q_mu)
        model.q_sqrt = gpflow.Parameter(new_q_sqrt, transform=gpflow.utilities.triangular())
