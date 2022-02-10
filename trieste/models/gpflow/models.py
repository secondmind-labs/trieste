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

from typing import Any, Optional, Union

import gpflow
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.base import (
    DType,
    Prior,
    PriorOn,
    TensorData,
    Transform,
    _cast_to_dtype,
    _validate_unconstrained_value,
)
from gpflow.conditionals.util import sample_mvn
from gpflow.inducing_variables import InducingPoints, InducingVariables
from gpflow.models import GPR, SGPR, SVGP, VGP
from gpflow.utilities import multiple_assign, read_values
from gpflow.utilities.ops import leading_transpose
from tensorflow_probability.python.util import TransformedVariable

from ...data import Dataset
from ...types import TensorType
from ...utils import DEFAULTS, jit
from ..interfaces import (
    FastUpdateModel,
    HasTrajectorySampler,
    SupportsInternalData,
    TrainableProbabilisticModel,
    TrajectorySampler,
)
from ..optimizer import BatchOptimizer, Optimizer
from .interface import GPflowPredictor, SupportsCovarianceBetweenPoints
from .sampler import RandomFourierFeatureTrajectorySampler
from .utils import (
    assert_data_is_compatible,
    check_optimizer,
    randomize_hyperparameters,
    squeeze_hyperparameters,
)


class GaussianProcessRegression(
    GPflowPredictor,
    TrainableProbabilisticModel,
    FastUpdateModel,
    SupportsCovarianceBetweenPoints,
    SupportsInternalData,
    HasTrajectorySampler,
):
    """
    A :class:`TrainableProbabilisticModel` wrapper for a GPflow :class:`~gpflow.models.GPR`
    or :class:`~gpflow.models.SGPR`.
    """

    def __init__(
        self,
        model: GPR | SGPR,
        optimizer: Optimizer | None = None,
        num_kernel_samples: int = 10,
        num_rff_features: int = 1000,
    ):
        """
        :param model: The GPflow model to wrap.
        :param optimizer: The optimizer with which to train the model. Defaults to
            :class:`~trieste.models.optimizer.Optimizer` with :class:`~gpflow.optimizers.Scipy`.
        :param num_kernel_samples: Number of randomly sampled kernels (for each kernel parameter) to
            evaluate before beginning model optimization. Therefore, for a kernel with `p`
            (vector-valued) parameters, we evaluate `p * num_kernel_samples` kernels.
        :param num_rff_features: The number of random Foruier features used to approximate the
            kernel when calling :meth:`trajectory_sampler`. We use a default of 1000 as it
            typically perfoms well for a wide range of kernels. Note that very smooth
            kernels (e.g. RBF) can be well-approximated with fewer features.
        """
        super().__init__(optimizer)
        self._model = model

        check_optimizer(self.optimizer)

        if num_kernel_samples <= 0:
            raise ValueError(
                f"num_kernel_samples must be greater or equal to zero but got {num_kernel_samples}."
            )
        self._num_kernel_samples = num_kernel_samples

        if num_rff_features <= 0:
            raise ValueError(
                f"num_rff_features must be greater or equal to zero but got {num_rff_features}."
            )
        self._num_rff_features = num_rff_features

        self._ensure_variable_model_data()

    def __repr__(self) -> str:
        """"""
        return f"GaussianProcessRegression({self.model!r}, {self.optimizer!r})"

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

        Note that query_points_2 must be a rank 2 tensor, but query_points_1 can
        have leading dimensions.

        :param query_points_1: Set of query points with shape [..., N, D]
        :param query_points_2: Sets of query points with shape [M, D]
        :return: Covariance matrix between the sets of query points with shape [..., L, N, M]
            (L being the number of latent GPs = number of output dimensions)
        """
        if isinstance(self.model, SGPR):
            raise NotImplementedError("Covariance between points is not supported for SGPR.")

        tf.debugging.assert_shapes(
            [(query_points_1, [..., "N", "D"]), (query_points_2, ["M", "D"])]
        )

        x = self.model.data[0].value()
        num_data = tf.shape(x)[0]
        s = tf.linalg.diag(tf.fill([num_data], self.model.likelihood.variance))

        K = self.model.kernel(x)  # [num_data, num_data] or [L, num_data, num_data]
        Kx1 = self.model.kernel(query_points_1, x)  # [..., N, num_data] or [..., L, N, num_data]
        Kx2 = self.model.kernel(x, query_points_2)  # [num_data, M] or [L, num_data, M]
        K12 = self.model.kernel(query_points_1, query_points_2)  # [..., N, M] or [..., L, N, M]

        if len(tf.shape(K)) == 2:
            # if single output GPR, the kernel does not return the latent dimension so
            # we add it back here
            K = tf.expand_dims(K, -3)
            Kx1 = tf.expand_dims(Kx1, -3)
            Kx2 = tf.expand_dims(Kx2, -3)
            K12 = tf.expand_dims(K12, -3)
        elif len(tf.shape(K)) > 3:
            raise NotImplementedError(
                "Covariance between points is not supported "
                "for kernels of type "
                f"{type(self.model.kernel)}."
            )

        L = tf.linalg.cholesky(K + s)  # [L, num_data, num_data]

        Kx1 = leading_transpose(Kx1, [..., -1, -2])  # [..., L, num_data, N]
        Linv_Kx1 = tf.linalg.triangular_solve(L, Kx1)  # [..., L, num_data, N]
        Linv_Kx2 = tf.linalg.triangular_solve(L, Kx2)  # [L, num_data, M]

        # The line below is just A^T*B over the last 2 dimensions.
        cov = K12 - tf.einsum("...lji,ljk->...lik", Linv_Kx1, Linv_Kx2)  # [..., L, N, M]

        num_latent = self.model.num_latent_gps
        if cov.shape[-3] == 1 and num_latent > 1:
            # For multioutput GPR with shared kernel, we need to duplicate cov
            # for each output
            cov = tf.repeat(cov, num_latent, axis=-3)

        tf.debugging.assert_shapes(
            [
                (query_points_1, [..., "N", "D"]),
                (query_points_2, ["M", "D"]),
                (cov, [..., "L", "N", "M"]),
            ]
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

    def trajectory_sampler(self) -> TrajectorySampler[GaussianProcessRegression]:
        """
        Return a trajectory sampler. For :class:`GaussianProcessRegression`, we build
        trajectories using a random Fourier feature approximation.

        :return: The trajectory sampler.
        """

        return RandomFourierFeatureTrajectorySampler(self, self._num_rff_features)

    def get_internal_data(self) -> Dataset:
        """
        Return the model's training data.

        :return: The model's training data.
        """
        return Dataset(self.model.data[0].value(), self.model.data[1].value())

    def conditional_predict_f(
        self, query_points: TensorType, additional_data: Dataset
    ) -> tuple[TensorType, TensorType]:
        """
        Returns the marginal GP distribution at query_points conditioned on both the model
        and some additional data, using exact formula. See :cite:`chevalier2014corrected`
        (eqs. 8-10) for details.

        :param query_points: Set of query points with shape [M, D]
        :param additional_data: Dataset with query_points with shape [..., N, D] and observations
                 with shape [..., N, L]
        :return: mean_qp_new: predictive mean at query_points, with shape [..., M, L],
                 and var_qp_new: predictive variance at query_points, with shape [..., M, L]
        """

        tf.debugging.assert_shapes(
            [
                (additional_data.query_points, [..., "N", "D"]),
                (additional_data.observations, [..., "N", "L"]),
                (query_points, ["M", "D"]),
            ],
            message="additional_data must have query_points with shape [..., N, D]"
            " and observations with shape [..., N, L], and query_points "
            "should have shape [M, D]",
        )

        if isinstance(self.model, SGPR):
            raise NotImplementedError("Conditional predict f is not supported for SGPR.")

        mean_add, cov_add = self.model.predict_f(
            additional_data.query_points, full_cov=True
        )  # [..., N, L], [..., L, N, N]
        mean_qp, var_qp = self.model.predict_f(query_points, full_cov=False)  # [M, L], [M, L]

        cov_cross = self.covariance_between_points(
            additional_data.query_points, query_points
        )  # [..., L, N, M]

        cov_shape = tf.shape(cov_add)
        noise = self.model.likelihood.variance * tf.eye(
            cov_shape[-2], batch_shape=cov_shape[:-2], dtype=cov_add.dtype
        )
        L_add = tf.linalg.cholesky(cov_add + noise)  # [..., L, N, N]
        A = tf.linalg.triangular_solve(L_add, cov_cross, lower=True)  # [..., L, N, M]
        var_qp_new = var_qp - leading_transpose(
            tf.reduce_sum(A ** 2, axis=-2), [..., -1, -2]
        )  # [..., M, L]

        mean_add_diff = additional_data.observations - mean_add  # [..., N, L]
        mean_add_diff = leading_transpose(mean_add_diff, [..., -1, -2])[..., None]  # [..., L, N, 1]
        AM = tf.linalg.triangular_solve(L_add, mean_add_diff)  # [..., L, N, 1]

        mean_qp_new = mean_qp + leading_transpose(
            (tf.matmul(A, AM, transpose_a=True)[..., 0]), [..., -1, -2]
        )  # [..., M, L]

        tf.debugging.assert_shapes(
            [
                (additional_data.observations, [..., "N", "L"]),
                (query_points, ["M", "D"]),
                (mean_qp_new, [..., "M", "L"]),
                (var_qp_new, [..., "M", "L"]),
            ],
            message="received unexpected shapes computing conditional_predict_f,"
            "check model kernel structure?",
        )

        return mean_qp_new, var_qp_new

    def conditional_predict_joint(
        self, query_points: TensorType, additional_data: Dataset
    ) -> tuple[TensorType, TensorType]:
        """
        Predicts the joint GP distribution at query_points conditioned on both the model
        and some additional data, using exact formula. See :cite:`chevalier2014corrected`
        (eqs. 8-10) for details.

        :param query_points: Set of query points with shape [M, D]
        :param additional_data: Dataset with query_points with shape [..., N, D] and observations
                 with shape [..., N, L]
        :return: mean_qp_new: predictive mean at query_points, with shape [..., M, L],
                 and cov_qp_new: predictive covariance between query_points, with shape
                 [..., L, M, M]
        """

        tf.debugging.assert_shapes(
            [
                (additional_data.query_points, [..., "N", "D"]),
                (additional_data.observations, [..., "N", "L"]),
                (query_points, ["M", "D"]),
            ],
            message="additional_data must have query_points with shape [..., N, D]"
            " and observations with shape [..., N, L], and query_points "
            "should have shape [M, D]",
        )

        if isinstance(self.model, SGPR):
            raise NotImplementedError("Conditional predict f is not supported for SGPR.")

        leading_dims = tf.shape(additional_data.query_points)[:-2]  # [...]
        new_shape = tf.concat([leading_dims, tf.shape(query_points)], axis=0)  # [..., M, D]
        query_points_r = tf.broadcast_to(query_points, new_shape)  # [..., M, D]
        points = tf.concat([additional_data.query_points, query_points_r], axis=-2)  # [..., N+M, D]

        mean, cov = self.model.predict_f(points, full_cov=True)  # [..., N+M, L], [..., L, N+M, N+M]

        N = tf.shape(additional_data.query_points)[-2]

        mean_add = mean[..., :N, :]  # [..., N, L]
        mean_qp = mean[..., N:, :]  # [..., M, L]

        cov_add = cov[..., :N, :N]  # [..., L, N, N]
        cov_qp = cov[..., N:, N:]  # [..., L, M, M]
        cov_cross = cov[..., :N, N:]  # [..., L, N, M]

        cov_shape = tf.shape(cov_add)
        noise = self.model.likelihood.variance * tf.eye(
            cov_shape[-2], batch_shape=cov_shape[:-2], dtype=cov_add.dtype
        )
        L_add = tf.linalg.cholesky(cov_add + noise)  # [..., L, N, N]
        A = tf.linalg.triangular_solve(L_add, cov_cross, lower=True)  # [..., L, N, M]
        cov_qp_new = cov_qp - tf.matmul(A, A, transpose_a=True)  # [..., L, M, M]

        mean_add_diff = additional_data.observations - mean_add  # [..., N, L]
        mean_add_diff = leading_transpose(mean_add_diff, [..., -1, -2])[..., None]  # [..., L, N, 1]
        AM = tf.linalg.triangular_solve(L_add, mean_add_diff)  # [..., L, N, 1]
        mean_qp_new = mean_qp + leading_transpose(
            (tf.matmul(A, AM, transpose_a=True)[..., 0]), [..., -1, -2]
        )  # [..., M, L]

        tf.debugging.assert_shapes(
            [
                (additional_data.observations, [..., "N", "L"]),
                (query_points, ["M", "D"]),
                (mean_qp_new, [..., "M", "L"]),
                (cov_qp_new, [..., "L", "M", "M"]),
            ],
            message="received unexpected shapes computing conditional_predict_joint,"
            "check model kernel structure?",
        )

        return mean_qp_new, cov_qp_new

    def conditional_predict_f_sample(
        self, query_points: TensorType, additional_data: Dataset, num_samples: int
    ) -> TensorType:
        """
        Generates samples of the GP at query_points conditioned on both the model
        and some additional data.

        :param query_points: Set of query points with shape [M, D]
        :param additional_data: Dataset with query_points with shape [..., N, D] and observations
                 with shape [..., N, L]
        :param num_samples: number of samples
        :return: samples of f at query points, with shape [..., num_samples, M, L]
        """

        if isinstance(self.model, SGPR):
            raise NotImplementedError("Conditional predict y is not supported for SGPR.")

        mean_new, cov_new = self.conditional_predict_joint(query_points, additional_data)
        mean_for_sample = tf.linalg.adjoint(mean_new)  # [..., L, N]
        samples = sample_mvn(
            mean_for_sample, cov_new, full_cov=True, num_samples=num_samples
        )  # [..., (S), P, N]
        return tf.linalg.adjoint(samples)  # [..., (S), N, L]

    def conditional_predict_y(
        self, query_points: TensorType, additional_data: Dataset
    ) -> tuple[TensorType, TensorType]:
        """
        Generates samples of y from the GP at query_points conditioned on both the model
        and some additional data.

        :param query_points: Set of query points with shape [M, D]
        :param additional_data: Dataset with query_points with shape [..., N, D] and observations
                 with shape [..., N, L]
        :return: predictive variance at query_points, with shape [..., M, L],
                 and predictive variance at query_points, with shape [..., M, L]
        """

        if isinstance(self.model, SGPR):
            raise NotImplementedError("Conditional predict y is not supported for SGPR.")
        f_mean, f_var = self.conditional_predict_f(query_points, additional_data)
        return self.model.likelihood.predict_mean_and_var(f_mean, f_var)


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


class Parameter(gpflow.Parameter):  # type: ignore[misc]
    """A modified version of gpflow.Parameter that supports variable shapes."""

    def __init__(
        self,
        value: TensorData,
        *,
        transform: Optional[Transform] = None,
        prior: Optional[Prior] = None,
        prior_on: Union[str, PriorOn] = PriorOn.CONSTRAINED,
        trainable: bool = True,
        dtype: Optional[DType] = None,
        name: Optional[str] = None,
        transformed_shape: Any = None,
        pretransformed_shape: Any = None,
    ):
        """
        A copy of gpflow.Parameter's init but with additional shape arguments that are passed
        to a modified TransformedVariable.
        """
        if transform is None:
            transform = tfp.bijectors.Identity()

        value = _cast_to_dtype(value, dtype)
        _validate_unconstrained_value(value, transform, dtype)

        # An inlined, modified version of TransformedVariable's __init__ that also passes any shape
        # parameter specified in kwargs to the DeferredTensor parent's __init__.
        # (use same parameter names as TransformedVariable)
        bijector = transform
        initial_value = value

        for attr in {
            "forward",
            "forward_event_shape",
            "inverse",
            "inverse_event_shape",
            "name",
            "dtype",
        }:
            if not hasattr(bijector, attr):
                raise TypeError(
                    "Argument `bijector` missing required `Bijector` "
                    'attribute "{}".'.format(attr)
                )

        if callable(initial_value):
            initial_value = initial_value()
        initial_value = tf.convert_to_tensor(initial_value, dtype_hint=bijector.dtype, dtype=dtype)
        super(TransformedVariable, self).__init__(
            pretransformed_input=tf.Variable(
                initial_value=bijector.inverse(initial_value),
                name=name,
                dtype=dtype,
                shape=pretransformed_shape,
            ),
            transform_fn=bijector,
            shape=transformed_shape or initial_value.shape,
            name=bijector.name,
        )
        self._bijector = bijector

        self.prior = prior
        self.prior_on = prior_on


class SparseVariational(
    GPflowPredictor,
    TrainableProbabilisticModel,
    SupportsCovarianceBetweenPoints,
):
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

        check_optimizer(optimizer)

        # GPflow stores num_data as a number. However, since we want to be able to update it
        # without having to retrace the acquisition functions, put it in a Variable instead.
        # So that the elbo method doesn't fail we also need to turn it into a property.
        if not isinstance(self._model, NumDataPropertyMixin):

            class SVGPWrapper(type(self._model), NumDataPropertyMixin):  # type: ignore
                """A wrapper around GPFlow's SVGP class that stores num_data in a tf.Variable and
                exposes it as a property."""

            self._model._num_data = tf.Variable(model.num_data, trainable=False, dtype=tf.float64)
            self._model.__class__ = SVGPWrapper

    def __repr__(self) -> str:
        """"""
        return f"SparseVariational({self.model!r}, {self.optimizer!r})"

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

    def covariance_between_points(
        self, query_points_1: TensorType, query_points_2: TensorType
    ) -> TensorType:
        r"""
        Compute the posterior covariance between sets of query points.

        Note that query_points_2 must be a rank 2 tensor, but query_points_1 can
        have leading dimensions.

        :param query_points_1: Set of query points with shape [..., A, D]
        :param query_points_2: Sets of query points with shape [B, D]
        :return: Covariance matrix between the sets of query points with shape [..., L, A, B]
            (L being the number of latent GPs = number of output dimensions)
        """

        return _covariance_between_points_for_variational_models(
            kernel=self.get_kernel(),
            inducing_variable=self.model.inducing_variable,
            q_sqrt=self.model.q_sqrt,
            query_points_1=query_points_1,
            query_points_2=query_points_2,
            whiten=self.model.whiten,
        )


class VariationalGaussianProcess(
    GPflowPredictor,
    TrainableProbabilisticModel,
    SupportsInternalData,
    SupportsCovarianceBetweenPoints,
):
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

        check_optimizer(self.optimizer)

        if use_natgrads:
            if not isinstance(self.optimizer.optimizer, tf.optimizers.Optimizer):
                raise ValueError(
                    f"""
                    Natgrads can only be used with a BatchOptimizer wrapper using an instance of
                    tf.optimizers.Optimizer, however received {self.optimizer}.
                    """
                )
            natgrad_gamma = 0.1 if natgrad_gamma is None else natgrad_gamma
        else:
            if isinstance(self.optimizer.optimizer, tf.optimizers.Optimizer):
                raise ValueError(
                    f"""
                    If not using natgrads an Optimizer wrapper should be used with
                    gpflow.optimizers.Scipy, however received {self.optimizer}.
                    """
                )
            if natgrad_gamma is not None:
                raise ValueError(
                    """
                    natgrad_gamma is only to be specified when use_natgrads is True.
                    """
                )

        self._model = model
        self._use_natgrads = use_natgrads
        self._natgrad_gamma = natgrad_gamma
        self._ensure_variable_model_data()

    def _ensure_variable_model_data(self) -> None:
        # GPflow stores the data in Tensors. However, since we want to be able to update the data
        # without having to retrace the acquisition functions, put it in Variables instead.
        # Data has to be stored in variables with dynamic shape to allow for changes
        # Sometimes, for instance after serialization-deserialization, the shape can be overridden
        # Thus here we ensure data is stored in dynamic shape Variables

        if not all(isinstance(x, tf.Variable) and x.shape[0] is None for x in self._model.data):

            self._model.data = (
                tf.Variable(
                    self._model.data[0],
                    trainable=False,
                    shape=[None, *self._model.data[0].shape[1:]],
                ),
                tf.Variable(
                    self._model.data[1],
                    trainable=False,
                    shape=[None, *self._model.data[1].shape[1:]],
                ),
            )

            self._model.q_mu = Parameter(
                self._model.q_mu,
                transform=self._model.q_mu.bijector,
                prior=self._model.q_mu.prior,
                prior_on=self._model.q_mu.prior_on,
                dtype=self._model.q_mu.dtype,
                name=self._model.q_mu.unconstrained_variable.name,
                trainable=self._model.q_mu.trainable,
                transformed_shape=[None, *self._model.q_mu.shape[1:]],
                pretransformed_shape=[None, *self._model.q_mu.unconstrained_variable.shape[1:]],
            )
            self._model.q_sqrt = Parameter(
                self._model.q_sqrt,
                transform=self._model.q_sqrt.bijector,
                prior=self._model.q_sqrt.prior,
                prior_on=self._model.q_sqrt.prior_on,
                dtype=self._model.q_sqrt.dtype,
                name=self._model.q_sqrt.unconstrained_variable.name,
                trainable=self._model.q_sqrt.trainable,
                transformed_shape=[*self._model.q_sqrt.shape[:-2], None, None],
                pretransformed_shape=[*self._model.q_sqrt.unconstrained_variable.shape[:-1], None],
            )

        # GPflow stores num_data as a number. However, since we want to be able to update it
        # without having to retrace the acquisition functions, put it in a Variable instead.
        # So that the elbo method doesn't fail we also need to turn it into a property.
        if not isinstance(self._model, NumDataPropertyMixin):

            class VGPWrapper(type(self._model), NumDataPropertyMixin):  # type: ignore
                """A wrapper around GPFlow's VGP class that stores num_data in a tf.Variable and
                exposes it as a property."""

            self._model._num_data = tf.Variable(
                self._model.num_data or 0, trainable=False, dtype=tf.float64
            )
            self._model.__class__ = VGPWrapper

    def __repr__(self) -> str:
        """"""
        return f"VariationalGaussianProcess({self.model!r}, {self.optimizer!r})"

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
        self._ensure_variable_model_data()

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
        model.q_mu.assign(Parameter(new_q_mu))
        model.q_sqrt.assign(Parameter(new_q_sqrt, transform=gpflow.utilities.triangular()))

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

    def get_internal_data(self) -> Dataset:
        """
        Return the model's training data.

        :return: The model's training data.
        """
        return Dataset(self.model.data[0].value(), self.model.data[1].value())

    def covariance_between_points(
        self, query_points_1: TensorType, query_points_2: TensorType
    ) -> TensorType:
        r"""
        Compute the posterior covariance between sets of query points.

        Note that query_points_2 must be a rank 2 tensor, but query_points_1 can
        have leading dimensions.

        :param query_points_1: Set of query points with shape [..., A, D]
        :param query_points_2: Sets of query points with shape [B, D]
        :return: Covariance matrix between the sets of query points with shape [..., L, A, B]
            (L being the number of latent GPs = number of output dimensions)
        """

        return _covariance_between_points_for_variational_models(
            kernel=self.get_kernel(),
            inducing_variable=InducingPoints(self.model.data[0].value()),  # pass copy
            q_sqrt=self.model.q_sqrt,
            query_points_1=query_points_1,
            query_points_2=query_points_2,
            whiten=True,  # GPflow's VGP model is hard-coded to use the whitened representation
        )


def _covariance_between_points_for_variational_models(
    kernel: gpflow.kernels.Kernel,
    inducing_variable: InducingVariables,
    q_sqrt: TensorType,
    query_points_1: TensorType,
    query_points_2: TensorType,
    whiten: bool,
) -> TensorType:
    r"""
    Compute the posterior covariance between sets of query points.

    .. math:: \Sigma_{12} = K_{1x}BK_{x2} + K_{12} - K_{1x}K_{xx}^{-1}K_{x2}

    where :math:`B = K_{xx}^{-1}(q_{sqrt}q_{sqrt}^T)K_{xx}^{-1}`
    or :math:`B = L^{-1}(q_{sqrt}q_{sqrt}^T)(L^{-1})^T` if we are using
    a whitened representation in our variational approximation. Here
    :math:`L` is the Cholesky decomposition of :math:`K_{xx}`.
    See :cite:`titsias2009variational` for a derivation.

    Note that this function can also be applied to
    our :class:`VariationalGaussianProcess` models by passing in the training
    data rather than the locations of the inducing points.

    Although query_points_2 must be a rank 2 tensor, query_points_1 can
    have leading dimensions.

    :inducing points: The input locations chosen for our variational approximation.
    :q_sqrt: The Cholesky decomposition of the covariance matrix of our
        variational distribution.
    :param query_points_1: Set of query points with shape [..., A, D]
    :param query_points_2: Sets of query points with shape [B, D]
    :param whiten:  If True then use whitened representations.
    :return: Covariance matrix between the sets of query points with shape [..., L, A, B]
        (L being the number of latent GPs = number of output dimensions)
    """

    tf.debugging.assert_shapes([(query_points_1, [..., "A", "D"]), (query_points_2, ["B", "D"])])

    num_latent = q_sqrt.shape[0]

    K, Kx1, Kx2, K12 = _compute_kernel_blocks(
        kernel, inducing_variable, query_points_1, query_points_2, num_latent
    )

    L = tf.linalg.cholesky(K)  # [L, M, M]
    Linv_Kx1 = tf.linalg.triangular_solve(L, Kx1)  # [..., L, M, A]
    Linv_Kx2 = tf.linalg.triangular_solve(L, Kx2)  # [..., L, M, B]

    def _leading_mul(M_1: TensorType, M_2: TensorType, transpose_a: bool) -> TensorType:
        if transpose_a:  # The einsum below is just A^T*B over the last 2 dimensions.
            return tf.einsum("...lji,ljk->...lik", M_1, M_2)
        else:  # The einsum below is just A*B^T over the last 2 dimensions.
            return tf.einsum("...lij,lkj->...lik", M_1, M_2)

    if whiten:
        first_cov_term = _leading_mul(
            _leading_mul(Linv_Kx1, q_sqrt, transpose_a=True),  # [..., L, A, M]
            _leading_mul(Linv_Kx2, q_sqrt, transpose_a=True),  # [..., L, B, M]
            transpose_a=False,
        )  # [..., L, A, B]
    else:
        Linv_qsqrt = tf.linalg.triangular_solve(L, q_sqrt)  # [L, M, M]
        first_cov_term = _leading_mul(
            _leading_mul(Linv_Kx1, Linv_qsqrt, transpose_a=True),  # [..., L, A, M]
            _leading_mul(Linv_Kx2, Linv_qsqrt, transpose_a=True),  # [..., L, B, M]
            transpose_a=False,
        )  # [..., L, A, B]

    second_cov_term = K12  # [..., L, A, B]
    third_cov_term = _leading_mul(Linv_Kx1, Linv_Kx2, transpose_a=True)  # [..., L, A, B]
    cov = first_cov_term + second_cov_term - third_cov_term  # [..., L, A, B]

    tf.debugging.assert_shapes(
        [
            (query_points_1, [..., "N", "D"]),
            (query_points_2, ["M", "D"]),
            (cov, [..., "L", "N", "M"]),
        ]
    )
    return cov


def _compute_kernel_blocks(
    kernel: gpflow.kernels.Kernel,
    inducing_variable: InducingVariables,
    query_points_1: TensorType,
    query_points_2: TensorType,
    num_latent: int,
) -> tuple[TensorType, TensorType, TensorType, TensorType]:
    """
    Return all the prior covariances required to calculate posterior covariances for each latent
    Gaussian process, as specified by the `num_latent` input.

    This function returns the covariance between: `inducing_variables` and `query_points_1`;
    `inducing_variables` and `query_points_2`; `query_points_1` and `query_points_2`;
    `inducing_variables` and `inducing_variables`.

    The calculations are performed differently depending on the type of
    kernel (single output, separate independent multi-output or shared independent
    multi-output) and inducing variables (simple set, SharedIndependent or SeparateIndependent).

    Note that `num_latents` is only used when we use a single kernel for a multi-output model.
    """

    if type(inducing_variable) in [gpflow.inducing_variables.SharedIndependentInducingVariables]:
        inducing_points = inducing_variable.inducing_variable.Z
    elif type(inducing_variable) in [
        gpflow.inducing_variables.SeparateIndependentInducingVariables
    ]:
        inducing_points = [
            inducing_variable.Z for inducing_variable in inducing_variable.inducing_variables
        ]
    else:
        inducing_points = inducing_variable.Z

    if type(kernel) in [gpflow.kernels.SharedIndependent, gpflow.kernels.SeparateIndependent]:
        if type(inducing_points) == list:

            K = tf.concat(
                [ker(Z)[None, ...] for ker, Z in zip(kernel.kernels, inducing_points)], axis=0
            )
            Kx1 = tf.concat(
                [
                    ker(Z, query_points_1)[None, ...]
                    for ker, Z in zip(kernel.kernels, inducing_points)
                ],
                axis=0,
            )  # [..., L, M, A]
            Kx2 = tf.concat(
                [
                    ker(Z, query_points_2)[None, ...]
                    for ker, Z in zip(kernel.kernels, inducing_points)
                ],
                axis=0,
            )  # [L, M, B]
            K12 = tf.concat(
                [ker(query_points_1, query_points_2)[None, ...] for ker in kernel.kernels], axis=0
            )  # [L, M, B]
        else:
            K = kernel(inducing_points, full_cov=True, full_output_cov=False)  # [L, M, M]
            Kx1 = kernel(
                inducing_points, query_points_1, full_cov=True, full_output_cov=False
            )  # [..., L, M, A]
            Kx2 = kernel(
                inducing_points, query_points_2, full_cov=True, full_output_cov=False
            )  # [L, M, B]
            K12 = kernel(
                query_points_1, query_points_2, full_cov=True, full_output_cov=False
            )  # [..., L, A, B]
    else:  # simple calculations for the single output case
        K = kernel(inducing_points)  # [M, M]
        Kx1 = kernel(inducing_points, query_points_1)  # [..., M, A]
        Kx2 = kernel(inducing_points, query_points_2)  # [M, B]
        K12 = kernel(query_points_1, query_points_2)  # [..., A, B]

    if len(tf.shape(K)) == 2:  # if single kernel then repeat for all latent dimensions
        K = tf.repeat(tf.expand_dims(K, -3), num_latent, axis=-3)
        Kx1 = tf.repeat(tf.expand_dims(Kx1, -3), num_latent, axis=-3)
        Kx2 = tf.repeat(tf.expand_dims(Kx2, -3), num_latent, axis=-3)
        K12 = tf.repeat(tf.expand_dims(K12, -3), num_latent, axis=-3)
    elif len(tf.shape(K)) > 3:
        raise NotImplementedError(
            "Covariance between points is not supported " "for kernels of type " f"{type(kernel)}."
        )

    tf.debugging.assert_shapes(
        [
            (K, ["L", "M", "M"]),
            (Kx1, ["L", "M", "A"]),
            (Kx2, ["L", "M", "B"]),
            (K12, ["L", "A", "B"]),
        ]
    )

    return K, Kx1, Kx2, K12
