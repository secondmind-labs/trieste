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

import copy
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Optional, TypeVar

import gpflow
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.models import GPR, SGPR, SVGP, VGP, GPModel
from gpflow.utilities import multiple_assign, read_values

from ..data import Dataset
from ..type import TensorType
from ..utils import DEFAULTS, jit
from .optimizer import Optimizer, TFOptimizer


class ProbabilisticModel(ABC):
    """A probabilistic model."""

    @abstractmethod
    def predict(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        """
        Return the mean and variance of the independent marginal distributions at each point in
        ``query_points``.

        This is essentially a convenience method for :meth:`predict_joint`, where non-event
        dimensions of ``query_points`` are all interpreted as broadcasting dimensions instead of
        batch dimensions, and the covariance is squeezed to remove redundant nesting.

        :param query_points: The points at which to make predictions, of shape [..., D].
        :return: The mean and variance of the independent marginal distributions at each point in
            ``query_points``. For a predictive distribution with event shape E, the mean and
            variance will both have shape [...] + E.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_joint(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        """
        :param query_points: The points at which to make predictions, of shape [..., B, D].
        :return: The mean and covariance of the joint marginal distribution at each batch of points
            in ``query_points``. For a predictive distribution with event shape E, the mean will
            have shape [..., B] + E, and the covariance shape [...] + E + [B, B].
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, query_points: TensorType, num_samples: int) -> TensorType:
        """
        Return ``num_samples`` samples from the independent marginal distributions at
        ``query_points``.

        :param query_points: The points at which to sample, with shape [..., N, D].
        :param num_samples: The number of samples at each point.
        :return: The samples. For a predictive distribution with event shape E, this has shape
            [..., S, N] + E, where S is the number of samples.
        """
        raise NotImplementedError

    def predict_y(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        """
        Return the mean and variance of the independent marginal distributions at each point in
        ``query_points`` for the observations, including noise contributions.

        Note that this is not supported by all models.

        :param query_points: The points at which to make predictions, of shape [..., D].
        :return: The mean and variance of the independent marginal distributions at each point in
            ``query_points``. For a predictive distribution with event shape E, the mean and
            variance will both have shape [...] + E.
        """
        raise NotImplementedError(
            f"Model {self!r} does not support predicting observations, just the latent function"
        )

    def get_observation_noise(self) -> TensorType:
        """
        Return the variance of observation noise.

        Note that this is not supported by all models.

        :return: The observation noise.
        """
        raise NotImplementedError(f"Model {self!r} does not provide scalar observation noise")

    def get_kernel(self) -> gpflow.kernels.Kernel:
        """
        Return the kernel of the model.

        :return: The kernel.
        """
        raise NotImplementedError("Model {self!r} does not provide observation noise")


class TrainableProbabilisticModel(ProbabilisticModel):
    """A trainable probabilistic model."""

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


class ModelStack(TrainableProbabilisticModel):
    r"""
    A :class:`ModelStack` is a wrapper around a number of :class:`TrainableProbabilisticModel`\ s.
    It combines the outputs of each model for predictions and sampling, and delegates training data
    to each model for updates and optimization.

    **Note:** Only supports vector outputs (i.e. with event shape [E]). Outputs for any two models
    are assumed independent. Each model may itself be single- or multi-output, and any one
    multi-output model may have dependence between its outputs. When we speak of *event size* in
    this class, we mean the output dimension for a given :class:`TrainableProbabilisticModel`,
    whether that is the :class:`ModelStack` itself, or one of the subsidiary
    :class:`TrainableProbabilisticModel`\ s within the :class:`ModelStack`. Of course, the event
    size for a :class:`ModelStack` will be the sum of the event sizes of each subsidiary model.
    """

    def __init__(
        self,
        model_with_event_size: tuple[TrainableProbabilisticModel, int],
        *models_with_event_sizes: tuple[TrainableProbabilisticModel, int],
    ):
        r"""
        The order of individual models specified at :meth:`__init__` determines the order of the
        :class:`ModelStack` output dimensions.

        :param model_with_event_size: The first model, and the size of its output events.
            **Note:** This is a separate parameter to ``models_with_event_sizes`` simply so that the
            method signature requires at least one model. It is not treated specially.
        :param \*models_with_event_sizes: The other models, and sizes of their output events.
        """
        super().__init__()
        self._models, self._event_sizes = zip(*(model_with_event_size,) + models_with_event_sizes)

    def predict(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        r"""
        :param query_points: The points at which to make predictions, of shape [..., D].
        :return: The predictions from all the wrapped models, concatenated along the event axis in
            the same order as they appear in :meth:`__init__`. If the wrapped models have predictive
            distributions with event shapes [:math:`E_i`], the mean and variance will both have
            shape [..., :math:`\sum_i E_i`].
        """
        means, vars_ = zip(*[model.predict(query_points) for model in self._models])
        return tf.concat(means, axis=-1), tf.concat(vars_, axis=-1)

    def predict_joint(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        r"""
        :param query_points: The points at which to make predictions, of shape [..., B, D].
        :return: The predictions from all the wrapped models, concatenated along the event axis in
            the same order as they appear in :meth:`__init__`. If the wrapped models have predictive
            distributions with event shapes [:math:`E_i`], the mean will have shape
            [..., B, :math:`\sum_i E_i`], and the covariance shape
            [..., :math:`\sum_i E_i`, B, B].
        """
        means, covs = zip(*[model.predict_joint(query_points) for model in self._models])
        return tf.concat(means, axis=-1), tf.concat(covs, axis=-3)

    def sample(self, query_points: TensorType, num_samples: int) -> TensorType:
        r"""
        :param query_points: The points at which to sample, with shape [..., N, D].
        :param num_samples: The number of samples at each point.
        :return: The samples from all the wrapped models, concatenated along the event axis. For
            wrapped models with predictive distributions with event shapes [:math:`E_i`], this has
            shape [..., S, N, :math:`\sum_i E_i`], where S is the number of samples.
        """
        samples = [model.sample(query_points, num_samples) for model in self._models]
        return tf.concat(samples, axis=-1)

    def predict_y(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        r"""
        :param query_points: The points at which to make predictions, of shape [..., D].
        :return: The predictions from all the wrapped models, concatenated along the event axis in
            the same order as they appear in :meth:`__init__`. If the wrapped models have predictive
            distributions with event shapes [:math:`E_i`], the mean and variance will both have
            shape [..., :math:`\sum_i E_i`].
        :raise NotImplementedError: If any of the models don't implement predict_y.
        """
        means, vars_ = zip(*[model.predict_y(query_points) for model in self._models])
        return tf.concat(means, axis=-1), tf.concat(vars_, axis=-1)

    def update(self, dataset: Dataset) -> None:
        """
        Update all the wrapped models on their corresponding data. The data for each model is
        extracted by splitting the observations in ``dataset`` along the event axis according to the
        event sizes specified at :meth:`__init__`.

        :param dataset: The query points and observations for *all* the wrapped models.
        """
        observations = tf.split(dataset.observations, self._event_sizes, axis=-1)

        for model, obs in zip(self._models, observations):
            model.update(Dataset(dataset.query_points, obs))

    def optimize(self, dataset: Dataset) -> None:
        """
        Optimize all the wrapped models on their corresponding data. The data for each model is
        extracted by splitting the observations in ``dataset`` along the event axis according to the
        event sizes specified at :meth:`__init__`.

        :param dataset: The query points and observations for *all* the wrapped models.
        """
        observations = tf.split(dataset.observations, self._event_sizes, axis=-1)

        for model, obs in zip(self._models, observations):
            model.optimize(Dataset(dataset.query_points, obs))


M = TypeVar("M", bound=tf.Module)
""" A type variable bound to :class:`tf.Module`. """


def module_deepcopy(self: M, memo: dict[int, object]) -> M:
    r"""
    This function provides a workaround for `a bug`_ in TensorFlow Probability (fixed in `version
    0.12`_) where a :class:`tf.Module` cannot be deep-copied if it has
    :class:`tfp.bijectors.Bijector` instances on it. The function can be used to directly copy an
    object ``self`` as e.g. ``module_deepcopy(self, {})``, but it is perhaps more useful as an
    implemention for :meth:`__deepcopy__` on classes, where it can be used as follows:

    .. _a bug: https://github.com/tensorflow/probability/issues/547
    .. _version 0.12: https://github.com/tensorflow/probability/releases/tag/v0.12.1

    .. testsetup:: *

        >>> import tensorflow_probability as tfp

    >>> class Foo(tf.Module):
    ...     example_bijector = tfp.bijectors.Exp()
    ...
    ...     __deepcopy__ = module_deepcopy

    Classes with this method can be deep-copied even if they contain
    :class:`tfp.bijectors.Bijector`\ s.

    :param self: The object to copy.
    :param memo: References to existing deep-copied objects (by object :func:`id`).
    :return: A deep-copy of ``self``.
    """
    gpflow.utilities.reset_cache_bijectors(self)

    new = self.__new__(type(self))
    memo[id(self)] = new

    for name, value in self.__dict__.items():
        setattr(new, name, copy.deepcopy(value, memo))

    return new


class GPflowPredictor(ProbabilisticModel, tf.Module, ABC):
    """A trainable wrapper for a GPflow Gaussian process model."""

    def __init__(self, optimizer: Optimizer | None = None):
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
        """The optimizer with which to train the model."""
        return self._optimizer

    @property
    @abstractmethod
    def model(self) -> GPModel:
        """The underlying GPflow model."""

    def predict(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        return self.model.predict_f(query_points)

    def predict_joint(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        return self.model.predict_f(query_points, full_cov=True)

    def sample(self, query_points: TensorType, num_samples: int) -> TensorType:
        return self.model.predict_f_samples(query_points, num_samples)

    def predict_y(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        return self.model.predict_y(query_points)

    def get_kernel(self) -> gpflow.kernels.Kernel:
        """
        Return the kernel of the model.

        :return: The kernel.
        """
        return self.model.kernel

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

    def optimize(self, dataset: Dataset) -> None:
        """
        Optimize the model with the specified `dataset`.

        :param dataset: The data with which to optimize the `model`.
        """
        self.optimizer.optimize(self.model, dataset)

    __deepcopy__ = module_deepcopy


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

    def __repr__(self) -> str:
        """"""
        return f"GaussianProcessRegression({self._model!r}, {self.optimizer!r})"

    @property
    def model(self) -> GPR | SGPR:
        return self._model

    def update(self, dataset: Dataset) -> None:
        x, y = self.model.data

        _assert_data_is_compatible(dataset, Dataset(x, y))

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

    def find_best_model_initialization(self, num_kernel_samples) -> None:
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
        _assert_data_is_compatible(dataset, self._data)

        self._data = dataset

        num_data = dataset.query_points.shape[0]
        self.model.num_data = num_data


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
            an :class:`~trieste.models.optimizer.Optimizer` optimizer.
        :natgrad_gamma: Gamma parameter for the natural gradient optimizer.
        :raise ValueError (or InvalidArgumentError): If ``model``'s :attr:`q_sqrt` is not rank 3
            or if attempting to combine natural gradients with a :class:`~gpflow.optimizers.Scipy`
            optimizer.
        """
        tf.debugging.assert_rank(model.q_sqrt, 3)
        super().__init__(optimizer)
        self._model = model

        if use_natgrads:
            if not isinstance(self._optimizer, TFOptimizer):
                raise ValueError(
                    f"""
                    Natgrads can only be used alongside an optimizer from tf.optimizers however
                    received f{self._optimizer}
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

        _assert_data_is_compatible(dataset, Dataset(*model.data))

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
            def perfom_optimization_step() -> None:  # alternate with natgrad optimizations
                natgrad_optimizer.minimize(loss_fn, variational_params)
                base_optimizer.optimizer.minimize(
                    loss_fn, model_params, **base_optimizer.minimize_args
                )

            for _ in range(base_optimizer.max_iter):  # type: ignore
                perfom_optimization_step()

            gpflow.set_trainable(model.q_mu, True)  # revert varitional params to trainable
            gpflow.set_trainable(model.q_sqrt, True)

        else:
            self.optimizer.optimize(model, dataset)


supported_models: dict[Any, Callable[[Any, Optimizer], TrainableProbabilisticModel]] = {
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


def randomize_hyperparameters(object: gpflow.Module) -> None:
    """
    Sets hyperparameters to random samples from their constrained domains or (if not constraints
    are available) their prior distributions.

    :param object: Any gpflow Module.
    """
    for param in object.trainable_parameters:
        if isinstance(param.bijector, tfp.bijectors.Sigmoid):
            sample = tf.random.uniform(
                param.bijector.low.shape,
                minval=param.bijector.low,
                maxval=param.bijector.high,
                dtype=param.bijector.low.dtype,
            )
            param.assign(sample)
        elif param.prior is not None:
            param.assign(param.prior.sample())


def squeeze_hyperparameters(
    object: gpflow.Module, alpha: float = 1e-2, epsilon: float = 1e-7
) -> None:
    """
    Squeezes the parameters to be strictly inside their range defined by the Sigmoid,
    or strictly greater than the limit defined by the Shift+Softplus.
    This avoids having Inf unconstrained values when the parameters are exactly at the boundary.

    :param object: Any gpflow Module.
    :param alpha: the proportion of the range with which to squeeze for the Sigmoid case
    :param epsilon: the value with which to offset the shift for the Softplus case.
    :raise ValueError: If ``alpha`` is not in (0,1) or epsilon <= 0
    """

    if not (0 < alpha < 1):
        raise ValueError(f"squeeze factor alpha must be in (0, 1), found {alpha}")

    if not (0 < epsilon):
        raise ValueError(f"offset factor epsilon must be > 0, found {epsilon}")

    for param in object.trainable_parameters:
        if isinstance(param.bijector, tfp.bijectors.Sigmoid):
            delta = (param.bijector.high - param.bijector.low) * alpha
            squeezed_param = tf.math.minimum(param, param.bijector.high - delta)
            squeezed_param = tf.math.maximum(squeezed_param, param.bijector.low + delta)
            param.assign(squeezed_param)
        elif (
            isinstance(param.bijector, tfp.bijectors.Chain)
            and len(param.bijector.bijectors) == 2
            and isinstance(param.bijector.bijectors[0], tfp.bijectors.Shift)
            and isinstance(param.bijector.bijectors[1], tfp.bijectors.Softplus)
        ):
            if isinstance(param.bijector.bijectors[0], tfp.bijectors.Shift) and isinstance(
                param.bijector.bijectors[1], tfp.bijectors.Softplus
            ):
                low = param.bijector.bijectors[0].shift
                squeezed_param = tf.math.maximum(param, low + epsilon * tf.ones_like(param))
                param.assign(squeezed_param)
