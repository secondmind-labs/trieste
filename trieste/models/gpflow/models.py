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

from __future__ import annotations

from typing import Optional

import gpflow
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.models import GPR, SGPR, SVGP, VGP
from gpflow.utilities import multiple_assign, read_values

from ...data import Dataset
from ...types import TensorType
from ...utils import DEFAULTS, jit
from ..interfaces import TrainableProbabilisticModel
from ..optimizer import BatchOptimizer, Optimizer
from .interface import GPflowPredictor
from .utils import assert_data_is_compatible, randomize_hyperparameters, squeeze_hyperparameters


class GaussianProcessRegression(GPflowPredictor, TrainableProbabilisticModel):
    """
    A :class:`TrainableProbabilisticModel` wrapper for a GPflow :class:`~gpflow.models.GPR`
    or :class:`~gpflow.models.SGPR`.
    """

    def __init__(
        self, model: GPR | SGPR, optimizer: Optimizer | None = None, num_kernel_samples: int = 10
    ):
        """
        :param model: The GPflow model to wrap.
        :param optimizer: The optimizer with which to train the model. Defaults to
            :class:`~trieste.models.optimizer.Optimizer` with :class:`~gpflow.optimizers.Scipy`.
        :param num_kernel_samples: Number of randomly sampled kernels (for each kernel parameter) to
            evaluate before beginning model optimization. Therefore, for a kernel with `p`
            (vector-valued) parameters, we evaluate `p * num_kernel_samples` kernels.
        """
        super().__init__(optimizer)
        self._model = model

        if num_kernel_samples <= 0:
            raise ValueError(
                f"num_kernel_samples must be greater or equal to zero but got {num_kernel_samples}."
            )
        self._num_kernel_samples = num_kernel_samples

        self._ensure_variable_model_data()

    def __repr__(self) -> str:
        """"""
        return f"GaussianProcessRegression({self._model!r}, {self.optimizer!r})"

    @property
    def model(self) -> GPR | SGPR:
        return self._model

    def _ensure_variable_model_data(self) -> None:
        # GPflow stores the data in Tensors. However, since we want to be able to update the data
        # without having to retrace the acquisition functions, put it in Variables instead.
        # Data has to be stored in variables with dynamic shape to allow for changes
        # Sometimes, for instance after serialization-deserialization, the shape can be overridden
        # Thus here we ensure data is stored in dynamic shape Variables

        if all(isinstance(x, tf.Variable) and x.shape[0] is None for x in self._model.data):
            # both query points and observations are in right shape
            # nothing to do
            return

        self._model.data = (
            tf.Variable(
                self._model.data[0], trainable=False, shape=[None, *self._model.data[0].shape[1:]]
            ),
            tf.Variable(
                self._model.data[1], trainable=False, shape=[None, *self._model.data[1].shape[1:]]
            ),
        )

    def update(self, dataset: Dataset) -> None:
        self._ensure_variable_model_data()

        x, y = self.model.data[0].value(), self.model.data[1].value()

        assert_data_is_compatible(dataset, Dataset(x, y))

        if dataset.query_points.shape[-1] != x.shape[-1]:
            raise ValueError

        if dataset.observations.shape[-1] != y.shape[-1]:
            raise ValueError

        self.model.data[0].assign(dataset.query_points)
        self.model.data[1].assign(dataset.observations)

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
        if isinstance(self.model, SGPR):
            raise NotImplementedError("Covariance between points is not supported for SGPR.")

        tf.debugging.assert_shapes([(query_points_1, ["N", "D"]), (query_points_2, ["M", "D"])])

        x = self.model.data[0].value()
        num_data = tf.shape(x)[0]
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

    def optimize(self, dataset: Dataset) -> None:
        """
        Optimize the model with the specified `dataset`.

        For :class:`GaussianProcessRegression`, we (optionally) try multiple randomly sampled
        kernel parameter configurations as well as the configuration specified when initializing
        the kernel. The best configuration is used as the starting point for model optimization.

        For trainable parameters constrained to lie in a finite interval (through a sigmoid
        bijector), we begin model optimization from the best of a random sample from these
        parameters' acceptable domains.

        For trainable parameters without constraints but with priors, we begin model optimization
        from the best of a random sample from these parameters' priors.

        For trainable parameters with neither priors nor constraints, we begin optimization from
        their initial values.

        :param dataset: The data with which to optimize the `model`.
        """

        num_trainable_params_with_priors_or_constraints = tf.reduce_sum(
            [
                tf.size(param)
                for param in self.model.trainable_parameters
                if param.prior is not None or isinstance(param.bijector, tfp.bijectors.Sigmoid)
            ]
        )

        if (
            min(num_trainable_params_with_priors_or_constraints, self._num_kernel_samples) >= 1
        ):  # Find a promising kernel initialization
            self.find_best_model_initialization(
                self._num_kernel_samples * num_trainable_params_with_priors_or_constraints
            )

        self.optimizer.optimize(self.model, dataset)

    def find_best_model_initialization(self, num_kernel_samples: int) -> None:
        """
        Test `num_kernel_samples` models with sampled kernel parameters. The model's kernel
        parameters are then set to the sample achieving maximal likelihood.

        :param num_kernel_samples: Number of randomly sampled kernels to evaluate.
        """

        @tf.function
        def evaluate_loss_of_model_parameters() -> tf.Tensor:
            randomize_hyperparameters(self.model)
            return self.model.training_loss()

        squeeze_hyperparameters(self.model)
        current_best_parameters = read_values(self.model)
        min_loss = self.model.training_loss()

        for _ in tf.range(num_kernel_samples):
            try:
                train_loss = evaluate_loss_of_model_parameters()
            except tf.errors.InvalidArgumentError:  # allow badly specified kernel params
                train_loss = 1e100

            if train_loss < min_loss:  # only keep best kernel params
                min_loss = train_loss
                current_best_parameters = read_values(self.model)

        multiple_assign(self.model, current_best_parameters)


class NumDataPropertyMixin:
    """Mixin class for exposing num_data as a property, stored in a tf.Variable. This is to work
    around GPFlow storing it as a number, which prevents us from updating it without retracing.
    The property is required due to the way num_data is used in the model elbo methods.

    Note that this doesn't support a num_data value of None.

    This should be removed once GPFlow is updated to support Variables as num_data.
    """

    _num_data: tf.Variable

    @property
    def num_data(self) -> TensorType:
        return self._num_data.value()

    @num_data.setter
    def num_data(self, value: TensorType) -> None:
        self._num_data.assign(value)


class SVGPWrapper(SVGP, NumDataPropertyMixin):
    """A wrapper around GPFlow's SVGP class that stores num_data in a tf.Variable and exposes
    it as a property."""


class SparseVariational(GPflowPredictor, TrainableProbabilisticModel):
    """
    A :class:`TrainableProbabilisticModel` wrapper for a GPflow :class:`~gpflow.models.SVGP`.
    """

    def __init__(self, model: SVGP, optimizer: Optimizer | None = None):
        """
        :param model: The underlying GPflow sparse variational model.
        :param optimizer: The optimizer with which to train the model. Defaults to
            :class:`~trieste.models.optimizer.BatchOptimizer` with :class:`~tf.optimizers.Adam` with
            batch size 100.
        """

        if optimizer is None:
            optimizer = BatchOptimizer(tf.optimizers.Adam(), batch_size=100)

        super().__init__(optimizer)
        self._model = model

        # GPflow stores num_data as a number. However, since we want to be able to update it
        # without having to retrace the acquisition functions, put it in a Variable instead.
        # So that the elbo method doesn't fail we also need to turn it into a property.
        if not isinstance(self._model, SVGPWrapper):
            self._model._num_data = tf.Variable(model.num_data, trainable=False, dtype=tf.float64)
            self._model.__class__ = SVGPWrapper

    def __repr__(self) -> str:
        """"""
        return f"SparseVariational({self._model!r}, {self.optimizer!r})"

    @property
    def model(self) -> SVGP:
        return self._model

    def update(self, dataset: Dataset) -> None:
        # Hard-code asserts from _assert_data_is_compatible because model doesn't store dataset
        if dataset.query_points.shape[-1] != self.model.inducing_variable.Z.shape[-1]:
            raise ValueError(
                f"Shape {dataset.query_points.shape} of new query points is incompatible with"
                f" shape {self.model.inducing_variable.Z.shape} of existing query points."
                f" Trailing dimensions must match."
            )

        if dataset.observations.shape[-1] != self.model.q_mu.shape[-1]:
            raise ValueError(
                f"Shape {dataset.observations.shape} of new observations is incompatible with"
                f" shape {self.model.q_mu.shape} of existing observations. Trailing"
                f" dimensions must match."
            )

        num_data = dataset.query_points.shape[0]
        self.model.num_data = num_data


class VGPWrapper(VGP, NumDataPropertyMixin):
    """A wrapper around GPFlow's VGP class that stores num_data in a tf.Variable and exposes
    it as a property."""


class VariationalGaussianProcess(GPflowPredictor, TrainableProbabilisticModel):
    r"""
    A :class:`TrainableProbabilisticModel` wrapper for a GPflow :class:`~gpflow.models.VGP`.

    A Variational Gaussian Process (VGP) approximates the posterior of a GP
    using the multivariate Gaussian closest to the posterior of the GP by minimizing the
    KL divergence between approximated and exact posteriors. See :cite:`opper2009variational`
    for details.

    The VGP provides (approximate) GP modelling under non-Gaussian likelihoods, for example
    when fitting a classification model over binary data.

    A whitened representation and (optional) natural gradient steps are used to aid
    model optimization.
    """

    def __init__(
        self,
        model: VGP,
        optimizer: Optimizer | None = None,
        use_natgrads: bool = False,
        natgrad_gamma: Optional[float] = None,
    ):
        """
        :param model: The GPflow :class:`~gpflow.models.VGP`.
        :param optimizer: The optimizer with which to train the model. Defaults to
            :class:`~trieste.models.optimizer.Optimizer` with :class:`~gpflow.optimizers.Scipy`.
        :param use_natgrads: If True then alternate model optimization steps with natural
            gradient updates. Note that natural gradients requires
            a :class:`~trieste.models.optimizer.BatchOptimizer` wrapper with
            :class:`~tf.optimizers.Optimizer` optimizer.
        :natgrad_gamma: Gamma parameter for the natural gradient optimizer.
        :raise ValueError (or InvalidArgumentError): If ``model``'s :attr:`q_sqrt` is not rank 3
            or if attempting to combine natural gradients with a :class:`~gpflow.optimizers.Scipy`
            optimizer.
        """
        tf.debugging.assert_rank(model.q_sqrt, 3)

        if optimizer is None and not use_natgrads:
            optimizer = Optimizer(gpflow.optimizers.Scipy())
        elif optimizer is None and use_natgrads:
            optimizer = BatchOptimizer(tf.optimizers.Adam(), batch_size=100)

        super().__init__(optimizer)
        self._model = model

        if use_natgrads:
            if not isinstance(self._optimizer, BatchOptimizer) or not isinstance(
                self._optimizer.optimizer, tf.optimizers.Optimizer
            ):
                raise ValueError(
                    f"""
                    Natgrads can only be used with a BatchOptimizer wrapper using an instance of
                    tf.optimizers.Optimizer, however received f{self._optimizer}.
                    """
                )

            natgrad_gamma = 0.1 if natgrad_gamma is None else natgrad_gamma
        else:
            if natgrad_gamma is not None:
                raise ValueError(
                    """
                    natgrad_gamma is only to be specified when use_natgrads is True.
                    """
                )

        self._use_natgrads = use_natgrads
        self._natgrad_gamma = natgrad_gamma

        # GPflow stores num_data as a number. However, since we want to be able to update it
        # without having to retrace the acquisition functions, put it in a Variable instead.
        # So that the elbo method doesn't fail we also need to turn it into a property.
        if not isinstance(self._model, VGPWrapper):
            self._model._num_data = tf.Variable(
                model.num_data or 0, trainable=False, dtype=tf.float64
            )
            self._model.__class__ = VGPWrapper

        # GPflow stores the data in Tensors. However, since we want to be able to update the data
        # without having to retrace the acquisition functions, put it in Variables instead.
        self._model.data = (
            tf.Variable(
                self._model.data[0], trainable=False, shape=[None, *self._model.data[0].shape[1:]]
            ),
            tf.Variable(
                self._model.data[1], trainable=False, shape=[None, *self._model.data[1].shape[1:]]
            ),
        )

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
        :param jitter: The size of the jitter to use when stabilizing the Cholesky decomposition of
            the covariance matrix.
        """
        model = self.model

        x, y = self.model.data[0].value(), self.model.data[1].value()
        assert_data_is_compatible(dataset, Dataset(x, y))

        f_mu, f_cov = self.model.predict_f(dataset.query_points, full_cov=True)  # [N, L], [L, N, N]

        # GPflow's VGP model is hard-coded to use the whitened representation, i.e.
        # q_mu and q_sqrt parametrize q(v), and u = f(X) = L v, where L = cholesky(K(X, X))
        # Hence we need to back-transform from f_mu and f_cov to obtain the updated
        # new_q_mu and new_q_sqrt:
        Knn = model.kernel(dataset.query_points, full_cov=True)  # [N, N]
        jitter_mat = jitter * tf.eye(len(dataset), dtype=Knn.dtype)
        Lnn = tf.linalg.cholesky(Knn + jitter_mat)  # [N, N]
        new_q_mu = tf.linalg.triangular_solve(Lnn, f_mu)  # [N, L]
        tmp = tf.linalg.triangular_solve(Lnn[None], f_cov)  # [L, N, N], L⁻¹ f_cov
        S_v = tf.linalg.triangular_solve(Lnn[None], tf.linalg.matrix_transpose(tmp))  # [L, N, N]
        new_q_sqrt = tf.linalg.cholesky(S_v + jitter_mat)  # [L, N, N]

        model.data[0].assign(dataset.query_points)
        model.data[1].assign(dataset.observations)
        model.num_data = len(dataset)
        model.q_mu = gpflow.Parameter(new_q_mu)
        model.q_sqrt = gpflow.Parameter(new_q_sqrt, transform=gpflow.utilities.triangular())

    def optimize(self, dataset: Dataset) -> None:
        """
        :class:`VariationalGaussianProcess` has a custom `optimize` method that (optionally) permits
        alternating between standard optimization steps (for kernel parameters) and natural gradient
        steps for the variational parameters (`q_mu` and `q_sqrt`). See :cite:`salimbeni2018natural`
        for details. Using natural gradients can dramatically speed up model fitting, especially for
        ill-conditioned posteriors.

        If using natural gradients, our optimizer inherits the mini-batch behavior and number
        of optimization steps as the base optimizer specified when initializing
        the :class:`VariationalGaussianProcess`.
        """
        model = self.model

        if self._use_natgrads:  # optimize variational params with natgrad optimizer

            natgrad_optimizer = gpflow.optimizers.NaturalGradient(gamma=self._natgrad_gamma)
            base_optimizer = self.optimizer

            gpflow.set_trainable(model.q_mu, False)  # variational params optimized by natgrad
            gpflow.set_trainable(model.q_sqrt, False)
            variational_params = [(model.q_mu, model.q_sqrt)]
            model_params = model.trainable_variables

            loss_fn = base_optimizer.create_loss(model, dataset)

            @jit(apply=self.optimizer.compile)
            def perform_optimization_step() -> None:  # alternate with natgrad optimizations
                natgrad_optimizer.minimize(loss_fn, variational_params)
                base_optimizer.optimizer.minimize(
                    loss_fn, model_params, **base_optimizer.minimize_args
                )

            for _ in range(base_optimizer.max_iter):  # type: ignore
                perform_optimization_step()

            gpflow.set_trainable(model.q_mu, True)  # revert varitional params to trainable
            gpflow.set_trainable(model.q_sqrt, True)

        else:
            self.optimizer.optimize(model, dataset)
