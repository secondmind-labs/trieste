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

from typing import Optional, Sequence, Tuple, Union, cast

import gpflow
import tensorflow as tf
import tensorflow_probability as tfp
from check_shapes import inherit_check_shapes
from gpflow.conditionals.util import sample_mvn
from gpflow.inducing_variables import (
    SeparateIndependentInducingVariables,
    SharedIndependentInducingVariables,
)
from gpflow.keras import tf_keras
from gpflow.logdensities import multivariate_normal
from gpflow.models import GPR, SGPR, SVGP, VGP
from gpflow.models.vgp import update_vgp_data
from gpflow.utilities import add_noise_cov, is_variable, multiple_assign, read_values
from gpflow.utilities.ops import leading_transpose

from ...data import (
    Dataset,
    add_fidelity_column,
    check_and_extract_fidelity_query_points,
    split_dataset_by_fidelity,
)
from ...space import EncoderFunction
from ...types import TensorType
from ...utils import DEFAULTS, jit
from ...utils.misc import flatten_leading_dims
from ..interfaces import (
    EncodedFastUpdateModel,
    HasTrajectorySampler,
    SupportsCovarianceWithTopFidelity,
    SupportsGetInducingVariables,
    SupportsGetInternalData,
    SupportsPredictY,
    TrainableProbabilisticModel,
    TrajectorySampler,
)
from ..optimizer import BatchOptimizer, Optimizer, OptimizeResult
from .inducing_point_selectors import InducingPointSelector
from .interface import EncodedSupportsCovarianceBetweenPoints, GPflowPredictor
from .sampler import DecoupledTrajectorySampler, RandomFourierFeatureTrajectorySampler
from .utils import (
    _covariance_between_points_for_variational_models,
    _whiten_points,
    assert_data_is_compatible,
    check_optimizer,
    randomize_hyperparameters,
    squeeze_hyperparameters,
)


class GaussianProcessRegression(
    GPflowPredictor,
    EncodedFastUpdateModel,
    EncodedSupportsCovarianceBetweenPoints,
    SupportsGetInternalData,
    HasTrajectorySampler,
):
    """
    A :class:`TrainableProbabilisticModel` wrapper for a GPflow :class:`~gpflow.models.GPR`.

    As Bayesian optimization requires a large number of sequential predictions (i.e. when maximizing
    acquisition functions), rather than calling the model directly at prediction time we instead
    call the posterior objects built by these models. These posterior objects store the
    pre-computed Gram matrices, which can be reused to allow faster subsequent predictions. However,
    note that these posterior objects need to be updated whenever the underlying model is changed
    by calling :meth:`update_posterior_cache` (this
    happens automatically after calls to :meth:`update` or :math:`optimize`).
    """

    def __init__(
        self,
        model: GPR,
        optimizer: Optimizer | None = None,
        num_kernel_samples: int = 10,
        num_rff_features: int = 1000,
        use_decoupled_sampler: bool = True,
        encoder: EncoderFunction | None = None,
    ):
        """
        :param model: The GPflow model to wrap.
        :param optimizer: The optimizer with which to train the model. Defaults to
            :class:`~trieste.models.optimizer.Optimizer` with :class:`~gpflow.optimizers.Scipy`.
        :param num_kernel_samples: Number of randomly sampled kernels (for each kernel parameter) to
            evaluate before beginning model optimization. Therefore, for a kernel with `p`
            (vector-valued) parameters, we evaluate `p * num_kernel_samples` kernels.
        :param num_rff_features: The number of random Fourier features used to approximate the
            kernel when calling :meth:`trajectory_sampler`. We use a default of 1000 as it
            typically perfoms well for a wide range of kernels. Note that very smooth
            kernels (e.g. RBF) can be well-approximated with fewer features.
        :param use_decoupled_sampler: If True use a decoupled random Fourier feature sampler, else
            just use a random Fourier feature sampler. The decoupled sampler suffers less from
            overestimating variance and can typically get away with a lower num_rff_features.
        :param encoder: Optional encoder with which to transform query points before
            generating predictions.
        """
        super().__init__(optimizer, encoder)
        self._model = model

        check_optimizer(self.optimizer)

        if num_kernel_samples < 0:
            raise ValueError(
                f"num_kernel_samples must be greater or equal to zero but got {num_kernel_samples}."
            )
        self._num_kernel_samples = num_kernel_samples

        if num_rff_features <= 0:
            raise ValueError(
                f"num_rff_features must be greater than zero but got {num_rff_features}."
            )
        self._num_rff_features = num_rff_features
        self._use_decoupled_sampler = use_decoupled_sampler
        self._ensure_variable_model_data()
        self.create_posterior_cache()

    def __repr__(self) -> str:
        """"""
        return (
            f"GaussianProcessRegression({self.model!r}, {self.optimizer!r},"
            f"{self._num_kernel_samples!r}, {self._num_rff_features!r},"
            f"{self._use_decoupled_sampler!r})"
        )

    @property
    def model(self) -> GPR:
        return self._model

    def _ensure_variable_model_data(self) -> None:
        # GPflow stores the data in Tensors. However, since we want to be able to update the data
        # without having to retrace the acquisition functions, put it in Variables instead.
        # Data has to be stored in variables with dynamic shape to allow for changes
        # Sometimes, for instance after serialization-deserialization, the shape can be overridden
        # Thus here we ensure data is stored in dynamic shape Variables

        if all(is_variable(x) and x.shape[0] is None for x in self._model.data):
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

    def predict_y_encoded(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        f_mean, f_var = self.predict_encoded(query_points)
        return self.model.likelihood.predict_mean_and_var(query_points, f_mean, f_var)

    def update_encoded(self, dataset: Dataset) -> None:
        self._ensure_variable_model_data()

        x, y = self.model.data[0].value(), self.model.data[1].value()

        assert_data_is_compatible(dataset, Dataset(x, y))

        if dataset.query_points.shape[-1] != x.shape[-1]:
            raise ValueError

        if dataset.observations.shape[-1] != y.shape[-1]:
            raise ValueError

        self.model.data[0].assign(dataset.query_points)
        self.model.data[1].assign(dataset.observations)
        self.update_posterior_cache()

    def covariance_between_points_encoded(
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

    def optimize_encoded(self, dataset: Dataset) -> OptimizeResult:
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
        result = self.optimizer.optimize(self.model, dataset)
        self.update_posterior_cache()
        return result

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

        At the moment only models with single latent GP are supported.

        :return: The trajectory sampler.
        :raise NotImplementedError: If we try to use the
            sampler with a model that has more than one latent GP.
        """
        if self.model.num_latent_gps > 1:
            raise NotImplementedError(
                f"""
                Trajectory sampler does not currently support models with multiple latent
                GPs, however received a model with {self.model.num_latent_gps} latent GPs.
                """
            )

        if self._use_decoupled_sampler:
            return DecoupledTrajectorySampler(self, self._num_rff_features)
        else:
            return RandomFourierFeatureTrajectorySampler(self, self._num_rff_features)

    def get_internal_data(self) -> Dataset:
        """
        Return the model's training data.

        :return: The model's training data.
        """
        return Dataset(self.model.data[0], self.model.data[1])

    def conditional_predict_f_encoded(
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

        mean_add, cov_add = self.predict_joint_encoded(
            additional_data.query_points
        )  # [..., N, L], [..., L, N, N]
        mean_qp, var_qp = self.predict_encoded(query_points)  # [M, L], [M, L]

        cov_cross = self.covariance_between_points(
            additional_data.query_points, query_points
        )  # [..., L, N, M]

        cov_shape = tf.shape(cov_add)
        noise = self.get_observation_noise() * tf.eye(
            cov_shape[-2], batch_shape=cov_shape[:-2], dtype=cov_add.dtype
        )
        L_add = tf.linalg.cholesky(cov_add + noise)  # [..., L, N, N]
        A = tf.linalg.triangular_solve(L_add, cov_cross, lower=True)  # [..., L, N, M]
        var_qp_new = var_qp - leading_transpose(
            tf.reduce_sum(A**2, axis=-2), [..., -1, -2]
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

    def conditional_predict_joint_encoded(
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

        leading_dims = tf.shape(additional_data.query_points)[:-2]  # [...]
        new_shape = tf.concat([leading_dims, tf.shape(query_points)], axis=0)  # [..., M, D]
        query_points_r = tf.broadcast_to(query_points, new_shape)  # [..., M, D]
        points = tf.concat([additional_data.query_points, query_points_r], axis=-2)  # [..., N+M, D]

        mean, cov = self.predict_joint_encoded(points)  # [..., N+M, L], [..., L, N+M, N+M]

        N = tf.shape(additional_data.query_points)[-2]

        mean_add = mean[..., :N, :]  # [..., N, L]
        mean_qp = mean[..., N:, :]  # [..., M, L]

        cov_add = cov[..., :N, :N]  # [..., L, N, N]
        cov_qp = cov[..., N:, N:]  # [..., L, M, M]
        cov_cross = cov[..., :N, N:]  # [..., L, N, M]

        cov_shape = tf.shape(cov_add)
        noise = self.get_observation_noise() * tf.eye(
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

    def conditional_predict_f_sample_encoded(
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

        mean_new, cov_new = self.conditional_predict_joint(query_points, additional_data)
        mean_for_sample = tf.linalg.adjoint(mean_new)  # [..., L, N]
        samples = sample_mvn(
            mean_for_sample, cov_new, full_cov=True, num_samples=num_samples
        )  # [..., (S), P, N]
        return tf.linalg.adjoint(samples)  # [..., (S), N, L]

    def conditional_predict_y_encoded(
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
        f_mean, f_var = self.conditional_predict_f(query_points, additional_data)
        return self.model.likelihood.predict_mean_and_var(query_points, f_mean, f_var)


class SparseGaussianProcessRegression(
    GPflowPredictor,
    EncodedSupportsCovarianceBetweenPoints,
    SupportsGetInducingVariables,
    SupportsGetInternalData,
    HasTrajectorySampler,
):
    """
    A :class:`TrainableProbabilisticModel` wrapper for a GPflow :class:`~gpflow.models.SGPR`.
    At the moment we only support models with a single latent GP. This is due to ``compute_qu``
    method in :class:`~gpflow.models.SGPR` that is used for computing covariance between
    query points and trajectory sampling, which at the moment works only for single latent GP.

    Similarly to our :class:`GaussianProcessRegression`, our :class:`~gpflow.models.SGPR` wrapper
    directly calls the posterior objects built by these models at prediction
    time. These posterior objects store the pre-computed Gram matrices, which can be reused to allow
    faster subsequent predictions. However, note that these posterior objects need to be updated
    whenever the underlying model is changed  by calling :meth:`update_posterior_cache` (this
    happens automatically after calls to :meth:`update` or :math:`optimize`).
    """

    def __init__(
        self,
        model: SGPR,
        optimizer: Optimizer | None = None,
        num_rff_features: int = 1000,
        inducing_point_selector: Optional[
            InducingPointSelector[SparseGaussianProcessRegression]
        ] = None,
        encoder: EncoderFunction | None = None,
    ):
        """
        :param model: The GPflow model to wrap.
        :param optimizer: The optimizer with which to train the model. Defaults to
            :class:`~trieste.models.optimizer.Optimizer` with :class:`~gpflow.optimizers.Scipy`.
        :param num_rff_features: The number of random Fourier features used to approximate the
            kernel when calling :meth:`trajectory_sampler`. We use a default of 1000 as it
            typically perfoms well for a wide range of kernels. Note that very smooth
            kernels (e.g. RBF) can be well-approximated with fewer features.
        :param inducing_point_selector: The (optional) desired inducing point selector that
            will update the underlying GPflow SGPR model's inducing points as
            the optimization progresses.
        :raise NotImplementedError (or ValueError): If we try to use a model with invalid
            ``num_rff_features``, or an ``inducing_point_selector`` with a model
            that has more than one set of inducing points.
        :param encoder: Optional encoder with which to transform query points before
            generating predictions.
        """
        super().__init__(optimizer, encoder)
        self._model = model

        check_optimizer(self.optimizer)

        if num_rff_features <= 0:
            raise ValueError(
                f"num_rff_features must be greater or equal to zero but got {num_rff_features}."
            )
        self._num_rff_features = num_rff_features

        if isinstance(self.model.inducing_variable, SeparateIndependentInducingVariables):
            if inducing_point_selector is not None:
                raise NotImplementedError(
                    f"""
                    InducingPointSelectors only currently support models with a single set
                    of inducing points however received inducing points of
                    type {type(self.model.inducing_variable)}.
                    """
                )
        self._inducing_point_selector = inducing_point_selector

        self._ensure_variable_model_data()
        self.create_posterior_cache()

    def __repr__(self) -> str:
        """"""
        return (
            f"SparseGaussianProcessRegression({self.model!r}, {self.optimizer!r},"
            f"{self._num_rff_features!r}, {self._inducing_point_selector!r})"
        )

    @property
    def model(self) -> SGPR:
        return self._model

    @property
    def inducing_point_selector(
        self,
    ) -> Optional[InducingPointSelector[SparseGaussianProcessRegression]]:
        return self._inducing_point_selector

    def predict_y_encoded(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        f_mean, f_var = self.predict_encoded(query_points)
        return self.model.likelihood.predict_mean_and_var(query_points, f_mean, f_var)

    def _ensure_variable_model_data(self) -> None:
        # GPflow stores the data in Tensors. However, since we want to be able to update the data
        # without having to retrace the acquisition functions, put it in Variables instead.
        # Data has to be stored in variables with dynamic shape to allow for changes
        # Sometimes, for instance after serialization-deserialization, the shape can be overridden
        # Thus here we ensure data is stored in dynamic shape Variables
        if not all(is_variable(x) and x.shape[0] is None for x in self._model.data):
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

        if not is_variable(self._model.num_data):
            self._model.num_data = tf.Variable(self._model.num_data, trainable=False)

    def optimize_encoded(self, dataset: Dataset) -> OptimizeResult:
        """
        Optimize the model with the specified `dataset`.

        :param dataset: The data with which to optimize the `model`.
        """
        result = self.optimizer.optimize(self.model, dataset)
        self.update_posterior_cache()
        return result

    def update_encoded(self, dataset: Dataset) -> None:
        self._ensure_variable_model_data()

        x, y = self.model.data[0].value(), self.model.data[1].value()

        assert_data_is_compatible(dataset, Dataset(x, y))

        if dataset.query_points.shape[-1] != x.shape[-1]:
            raise ValueError

        if dataset.observations.shape[-1] != y.shape[-1]:
            raise ValueError

        self.model.data[0].assign(dataset.query_points)
        self.model.data[1].assign(dataset.observations)

        current_inducing_points, q_mu, _, _ = self.get_inducing_variables()

        if isinstance(current_inducing_points, list):
            inducing_points_trailing_dim = current_inducing_points[0].shape[-1]
        else:
            inducing_points_trailing_dim = current_inducing_points.shape[-1]

        if dataset.query_points.shape[-1] != inducing_points_trailing_dim:
            raise ValueError(
                f"Shape {dataset.query_points.shape} of new query points is incompatible with"
                f" shape {self.model.inducing_variable.Z.shape} of existing query points."
                f" Trailing dimensions must match."
            )

        if dataset.observations.shape[-1] != q_mu.shape[-1]:
            raise ValueError(
                f"Shape {dataset.observations.shape} of new observations is incompatible with"
                f" shape {self.model.q_mu.shape} of existing observations. Trailing"
                f" dimensions must match."
            )

        num_data = dataset.query_points.shape[0]
        self.model.num_data.assign(num_data)

        if self._inducing_point_selector is not None:
            new_inducing_points = self._inducing_point_selector.calculate_inducing_points(
                current_inducing_points, self, dataset
            )
            if not tf.reduce_all(
                tf.math.equal(
                    new_inducing_points,
                    current_inducing_points,
                )
            ):  # only bother updating if points actually change
                self._update_inducing_variables(new_inducing_points)

        self.update_posterior_cache()

    def _update_inducing_variables(self, new_inducing_points: TensorType) -> None:
        """
        When updating the inducing points of a model, we must also update the other
        inducing variables, i.e. `q_mu` and `q_sqrt` accordingly. The exact form of this update
        depends if we are using whitened representations of the inducing variables.
        See :meth:`_whiten_points` for details.

        :param new_inducing_points: The desired values for the new inducing points.
        :raise NotImplementedError: If we try to update the inducing variables of a model
            that has more than one set of inducing points.
        """

        if isinstance(new_inducing_points, list):
            raise NotImplementedError(
                f"""
                We do not currently support updating models with multiple sets of
                inducing points however received; {new_inducing_points}
                """
            )

        old_inducing_points, _, _, _ = self.get_inducing_variables()
        tf.assert_equal(
            tf.shape(old_inducing_points), tf.shape(new_inducing_points)
        )  # number of inducing points must not change

        if isinstance(self.model.inducing_variable, SharedIndependentInducingVariables):
            # gpflow says inducing_variable might be a ndarray; it won't
            cast(TensorType, self.model.inducing_variable.inducing_variable).Z.assign(
                new_inducing_points
            )  # [M, D]
        else:
            self.model.inducing_variable.Z.assign(new_inducing_points)  # [M, D]

    def get_inducing_variables(
        self,
    ) -> Tuple[Union[TensorType, list[TensorType]], TensorType, TensorType, bool]:
        """
        Return the model's inducing variables. The SGPR model does not have ``q_mu``, ``q_sqrt`` and
        ``whiten`` objects. We can use ``compute_qu`` method to obtain ``q_mu`` and ``q_sqrt``,
        while the SGPR model does not use the whitened representation. Note that at the moment
        ``compute_qu`` works only for single latent GP and returns ``q_sqrt`` in a shape that is
        inconsistent with the SVGP model (hence we need to do modify its shape).

        :return: The inducing points (i.e. locations of the inducing variables), as a Tensor or a
            list of Tensors (when the model has multiple inducing points); a tensor containing the
            variational mean ``q_mu``; a tensor containing the Cholesky decomposition of the
            variational covariance ``q_sqrt``; and a bool denoting if we are using whitened or
            non-whitened representations.
        :raise NotImplementedError: If the model has more than one latent GP.
        """
        if self.model.num_latent_gps > 1:
            raise NotImplementedError(
                f"""
                We do not currently support models with more than one latent GP,
                however received a model with {self.model.num_latent_gps} outputs.
                """
            )

        inducing_variable = self.model.inducing_variable

        if isinstance(inducing_variable, SharedIndependentInducingVariables):
            # gpflow says inducing_variable might be a ndarray; it won't
            inducing_points = cast(TensorType, inducing_variable.inducing_variable).Z  # [M, D]
        elif isinstance(inducing_variable, SeparateIndependentInducingVariables):
            inducing_points = [
                cast(TensorType, inducing_variable).Z
                for inducing_variable in inducing_variable.inducing_variables
            ]  # list of L [M, D] tensors
        else:
            inducing_points = inducing_variable.Z  # [M, D]

        q_mu, q_var = self.model.compute_qu()
        q_sqrt = tf.linalg.cholesky(q_var)
        q_sqrt = tf.expand_dims(q_sqrt, 0)
        whiten = False

        return inducing_points, q_mu, q_sqrt, whiten

    def covariance_between_points_encoded(
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
        inducing_points, _, q_sqrt, whiten = self.get_inducing_variables()

        return _covariance_between_points_for_variational_models(
            kernel=self.get_kernel(),
            inducing_points=inducing_points,
            q_sqrt=q_sqrt,
            query_points_1=query_points_1,
            query_points_2=query_points_2,
            whiten=whiten,
        )

    def trajectory_sampler(self) -> TrajectorySampler[SparseGaussianProcessRegression]:
        """
        Return a trajectory sampler. For :class:`SparseGaussianProcessRegression`, we build
        trajectories using a decoupled random Fourier feature approximation. Note that this
        is available only for single output models.

        At the moment only models with single latent GP are supported.

        :return: The trajectory sampler.
        :raise NotImplementedError: If we try to use the
            sampler with a model that has more than one latent GP.
        """
        if self.model.num_latent_gps > 1:
            raise NotImplementedError(
                f"""
                Trajectory sampler does not currently support models with multiple latent
                GPs, however received a model with {self.model.num_latent_gps} latent GPs.
                """
            )

        return DecoupledTrajectorySampler(self, self._num_rff_features)

    def get_internal_data(self) -> Dataset:
        """
        Return the model's training data.

        :return: The model's training data.
        """
        return Dataset(self.model.data[0], self.model.data[1])


class SparseVariational(
    GPflowPredictor,
    EncodedSupportsCovarianceBetweenPoints,
    SupportsGetInducingVariables,
    HasTrajectorySampler,
):
    """
    A :class:`TrainableProbabilisticModel` wrapper for a GPflow :class:`~gpflow.models.SVGP`.

    Similarly to our :class:`GaussianProcessRegression`, our :class:`~gpflow.models.SVGP` wrapper
    directly calls the posterior objects built by these models at prediction
    time. These posterior objects store the pre-computed Gram matrices, which can be reused to allow
    faster subsequent predictions. However, note that these posterior objects need to be updated
    whenever the underlying model is changed  by calling :meth:`update_posterior_cache` (this
    happens automatically after calls to :meth:`update` or :math:`optimize`).
    """

    def __init__(
        self,
        model: SVGP,
        optimizer: Optimizer | None = None,
        num_rff_features: int = 1000,
        inducing_point_selector: Optional[InducingPointSelector[SparseVariational]] = None,
        encoder: EncoderFunction | None = None,
    ):
        """
        :param model: The underlying GPflow sparse variational model.
        :param optimizer: The optimizer with which to train the model. Defaults to
            :class:`~trieste.models.optimizer.BatchOptimizer` with :class:`~tf.optimizers.Adam` with
            batch size 100.
        :param num_rff_features: The number of random Fourier features used to approximate the
            kernel when performing decoupled Thompson sampling through
            its :meth:`trajectory_sampler`. We use a default of 1000 as it typically
            perfoms well for a wide range of kernels. Note that very smooth kernels (e.g. RBF)
            can be well-approximated with fewer features.
        :param inducing_point_selector: The (optional) desired inducing_point_selector that
            will update the underlying GPflow sparse variational model's inducing points as
            the optimization progresses.
        :raise NotImplementedError: If we try to use an inducing_point_selector with a model
            that has more than one set of inducing points.
        :param encoder: Optional encoder with which to transform query points before
            generating predictions.
        """

        tf.debugging.assert_rank(
            model.q_sqrt, 3, "SparseVariational requires an SVGP model with q_diag=False."
        )

        if optimizer is None:
            optimizer = BatchOptimizer(tf_keras.optimizers.Adam(), batch_size=100, compile=True)

        super().__init__(optimizer, encoder)
        self._model = model

        if num_rff_features <= 0:
            raise ValueError(
                f"num_rff_features must be greater or equal to zero but got {num_rff_features}."
            )
        self._num_rff_features = num_rff_features

        check_optimizer(optimizer)

        if isinstance(self.model.inducing_variable, SeparateIndependentInducingVariables):
            if inducing_point_selector is not None:
                raise NotImplementedError(
                    f"""
                    InducingPointSelectors only currently support models with a single set
                    of inducing points however received inducing points of
                    type {type(self.model.inducing_variable)}.
                    """
                )

        self._inducing_point_selector = inducing_point_selector
        self._ensure_variable_model_data()
        self.create_posterior_cache()

    def _ensure_variable_model_data(self) -> None:
        # GPflow stores the data in Tensors. However, since we want to be able to update the data
        # without having to retrace the acquisition functions, put it in Variables instead.
        # Data has to be stored in variables with dynamic shape to allow for changes
        # Sometimes, for instance after serialization-deserialization, the shape can be overridden
        # Thus here we ensure data is stored in dynamic shape Variables
        if not is_variable(self._model.num_data):
            self._model.num_data = tf.Variable(self._model.num_data, trainable=False)

    def __repr__(self) -> str:
        """"""
        return (
            f"SparseVariational({self.model!r}, {self.optimizer!r},"
            f"{self._num_rff_features!r}, {self._inducing_point_selector!r})"
        )

    @property
    def model(self) -> SVGP:
        return self._model

    @property
    def inducing_point_selector(self) -> Optional[InducingPointSelector[SparseVariational]]:
        return self._inducing_point_selector

    def predict_y_encoded(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        f_mean, f_var = self.predict_encoded(query_points)
        return self.model.likelihood.predict_mean_and_var(query_points, f_mean, f_var)

    def update_encoded(self, dataset: Dataset) -> None:
        self._ensure_variable_model_data()

        # Hard-code asserts from _assert_data_is_compatible because model doesn't store dataset
        current_inducing_points, q_mu, _, _ = self.get_inducing_variables()

        if isinstance(current_inducing_points, list):
            inducing_points_trailing_dim = current_inducing_points[0].shape[-1]
        else:
            inducing_points_trailing_dim = current_inducing_points.shape[-1]

        if dataset.query_points.shape[-1] != inducing_points_trailing_dim:
            raise ValueError(
                f"Shape {dataset.query_points.shape} of new query points is incompatible with"
                f" shape {self.model.inducing_variable.Z.shape} of existing query points."
                f" Trailing dimensions must match."
            )

        if dataset.observations.shape[-1] != q_mu.shape[-1]:
            raise ValueError(
                f"Shape {dataset.observations.shape} of new observations is incompatible with"
                f" shape {self.model.q_mu.shape} of existing observations. Trailing"
                f" dimensions must match."
            )

        num_data = dataset.query_points.shape[0]
        assert self.model.num_data is not None
        self.model.num_data.assign(num_data)

        if self._inducing_point_selector is not None:
            new_inducing_points = self._inducing_point_selector.calculate_inducing_points(
                current_inducing_points, self, dataset
            )
            if not tf.reduce_all(
                tf.math.equal(
                    new_inducing_points,
                    current_inducing_points,
                )
            ):  # only bother updating if points actually change
                self._update_inducing_variables(new_inducing_points)
                self.update_posterior_cache()

    def optimize_encoded(self, dataset: Dataset) -> OptimizeResult:
        """
        Optimize the model with the specified `dataset`.

        :param dataset: The data with which to optimize the `model`.
        """
        result = self.optimizer.optimize(self.model, dataset)
        self.update_posterior_cache()
        return result

    def _update_inducing_variables(self, new_inducing_points: TensorType) -> None:
        """
        When updating the inducing points of a model, we must also update the other
        inducing variables, i.e. `q_mu` and `q_sqrt` accordingly. The exact form of this update
        depends if we are using whitened representations of the inducing variables.
        See :meth:`_whiten_points` for details.

        :param new_inducing_points: The desired values for the new inducing points.
        :raise NotImplementedError: If we try to update the inducing variables of a model
            that has more than one set of inducing points.
        """

        if isinstance(new_inducing_points, list):
            raise NotImplementedError(
                f"""
                We do not currently support updating models with multiple sets of
                inducing points however received; {new_inducing_points}
                """
            )

        old_inducing_points, _, _, whiten = self.get_inducing_variables()
        tf.assert_equal(
            tf.shape(old_inducing_points), tf.shape(new_inducing_points)
        )  # number of inducing points must not change

        if whiten:
            new_q_mu, new_q_sqrt = _whiten_points(self, new_inducing_points)
        else:
            new_q_mu, new_f_cov = self.predict_joint_encoded(
                new_inducing_points
            )  # [N, L], [L, N, N]
            new_q_mu -= self.model.mean_function(new_inducing_points)
            jitter_mat = DEFAULTS.JITTER * tf.eye(
                tf.shape(new_inducing_points)[0], dtype=new_f_cov.dtype
            )
            new_q_sqrt = tf.linalg.cholesky(new_f_cov + jitter_mat)

        self.model.q_mu.assign(new_q_mu)  # [N, L]
        self.model.q_sqrt.assign(new_q_sqrt)  # [L, N, N]

        if isinstance(self.model.inducing_variable, SharedIndependentInducingVariables):
            # gpflow says inducing_variable might be a ndarray; it won't
            cast(TensorType, self.model.inducing_variable.inducing_variable).Z.assign(
                new_inducing_points
            )  # [M, D]
        else:
            self.model.inducing_variable.Z.assign(new_inducing_points)  # [M, D]

    def get_inducing_variables(
        self,
    ) -> Tuple[Union[TensorType, list[TensorType]], TensorType, TensorType, bool]:
        """
        Return the model's inducing variables.

        :return: The inducing points (i.e. locations of the inducing variables), as a Tensor or a
            list of Tensors (when the model has multiple inducing points); A tensor containing the
            variational mean q_mu; a tensor containing the Cholesky decomposition of the variational
            covariance q_sqrt; and a bool denoting if we are using whitened or
            non-whitened representations.
        """
        inducing_variable = self.model.inducing_variable

        if isinstance(inducing_variable, SharedIndependentInducingVariables):
            # gpflow says inducing_variable might be a ndarray; it won't
            inducing_points = cast(TensorType, inducing_variable.inducing_variable).Z  # [M, D]
        elif isinstance(inducing_variable, SeparateIndependentInducingVariables):
            inducing_points = [
                cast(TensorType, inducing_variable).Z
                for inducing_variable in inducing_variable.inducing_variables
            ]  # list of L [M, D] tensors
        else:
            inducing_points = inducing_variable.Z  # [M, D]

        return inducing_points, self.model.q_mu, self.model.q_sqrt, self.model.whiten

    def covariance_between_points_encoded(
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
        inducing_points, _, q_sqrt, whiten = self.get_inducing_variables()

        return _covariance_between_points_for_variational_models(
            kernel=self.get_kernel(),
            inducing_points=inducing_points,
            q_sqrt=q_sqrt,
            query_points_1=query_points_1,
            query_points_2=query_points_2,
            whiten=whiten,
        )

    def trajectory_sampler(self) -> TrajectorySampler[SparseVariational]:
        """
        Return a trajectory sampler. For :class:`SparseVariational`, we build
        trajectories using a decoupled random Fourier feature approximation.

        :return: The trajectory sampler.
        """
        return DecoupledTrajectorySampler(self, self._num_rff_features)


class VariationalGaussianProcess(
    GPflowPredictor,
    EncodedSupportsCovarianceBetweenPoints,
    SupportsGetInducingVariables,
    HasTrajectorySampler,
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

    Similarly to our :class:`GaussianProcessRegression`, our :class:`~gpflow.models.VGP` wrapper
    directly calls the posterior objects built by these models at prediction
    time. These posterior objects store the pre-computed Gram matrices, which can be reused to allow
    faster subsequent predictions. However, note that these posterior objects need to be updated
    whenever the underlying model is changed  by calling :meth:`update_posterior_cache` (this
    happens automatically after calls to :meth:`update` or :math:`optimize`).
    """

    def __init__(
        self,
        model: VGP,
        optimizer: Optimizer | None = None,
        use_natgrads: bool = False,
        natgrad_gamma: Optional[float] = None,
        num_rff_features: int = 1000,
        encoder: EncoderFunction | None = None,
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
        :param num_rff_features: The number of random Fourier features used to approximate the
            kernel when performing decoupled Thompson sampling through
            its :meth:`trajectory_sampler`. We use a default of 1000 as it typically perfoms
            well for a wide range of kernels. Note that very smooth kernels (e.g. RBF) can
            be well-approximated with fewer features.
        :raise ValueError (or InvalidArgumentError): If ``model``'s :attr:`q_sqrt` is not rank 3
            or if attempting to combine natural gradients with a :class:`~gpflow.optimizers.Scipy`
            optimizer.
        :param encoder: Optional encoder with which to transform query points before
            generating predictions.
        """
        tf.debugging.assert_rank(model.q_sqrt, 3)

        if optimizer is None and not use_natgrads:
            optimizer = Optimizer(gpflow.optimizers.Scipy(), compile=True)
        elif optimizer is None and use_natgrads:
            optimizer = BatchOptimizer(tf_keras.optimizers.Adam(), batch_size=100, compile=True)

        super().__init__(optimizer, encoder)

        check_optimizer(self.optimizer)

        if use_natgrads:
            if not isinstance(self.optimizer.optimizer, tf_keras.optimizers.Optimizer):
                raise ValueError(
                    f"""
                    Natgrads can only be used with a BatchOptimizer wrapper using an instance of
                    tf.optimizers.Optimizer, however received {self.optimizer}.
                    """
                )
            natgrad_gamma = 0.1 if natgrad_gamma is None else natgrad_gamma
        else:
            if isinstance(self.optimizer.optimizer, tf_keras.optimizers.Optimizer):
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

        if num_rff_features <= 0:
            raise ValueError(
                f"num_rff_features must be greater or equal to zero but got {num_rff_features}."
            )
        self._num_rff_features = num_rff_features

        self._model = model
        self._use_natgrads = use_natgrads
        self._natgrad_gamma = natgrad_gamma
        self._ensure_variable_model_data()
        self.create_posterior_cache()

    def _ensure_variable_model_data(self) -> None:
        # GPflow stores the data in Tensors. However, since we want to be able to update the data
        # without having to retrace the acquisition functions, put it in Variables instead.
        # Data has to be stored in variables with dynamic shape to allow for changes
        # Sometimes, for instance after serialization-deserialization, the shape can be overridden
        # Thus here we ensure data is stored in dynamic shape Variables

        model = self.model
        if not all(isinstance(x, tf.Variable) and x.shape[0] is None for x in model.data):
            variable_data = (
                tf.Variable(
                    model.data[0],
                    trainable=False,
                    shape=[None, *model.data[0].shape[1:]],
                ),
                tf.Variable(
                    model.data[1],
                    trainable=False,
                    shape=[None, *model.data[1].shape[1:]],
                ),
            )

            # reinitialise the model so that the underlying Parameters have the right shape
            # and then reassign the original values
            old_q_mu = model.q_mu
            old_q_sqrt = model.q_sqrt
            model.__init__(  # type: ignore[misc]
                variable_data,
                model.kernel,
                model.likelihood,
                model.mean_function,
                model.num_latent_gps,
            )
            model.q_mu.assign(old_q_mu)
            model.q_sqrt.assign(old_q_sqrt)

    def __repr__(self) -> str:
        """"""
        return (
            f"VariationalGaussianProcess({self.model!r}, {self.optimizer!r})"
            f"{self._use_natgrads!r}, {self._natgrad_gamma!r}, {self._num_rff_features!r})"
        )

    @property
    def model(self) -> VGP:
        return self._model

    def predict_y_encoded(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        f_mean, f_var = self.predict_encoded(query_points)
        return self.model.likelihood.predict_mean_and_var(query_points, f_mean, f_var)

    def update_encoded(self, dataset: Dataset, *, jitter: float = DEFAULTS.JITTER) -> None:
        """
        Update the model given the specified ``dataset``. Does not train the model.

        :param dataset: The data with which to update the model.
        :param jitter: The size of the jitter to use when stabilizing the Cholesky decomposition of
            the covariance matrix.
        """
        self._ensure_variable_model_data()
        update_vgp_data(self.model, (dataset.query_points, dataset.observations))
        self.update_posterior_cache()

    def optimize_encoded(self, dataset: Dataset) -> Optional[OptimizeResult]:
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

            gpflow.set_trainable(model.q_mu, True)  # revert variational params to trainable
            gpflow.set_trainable(model.q_sqrt, True)
            result = None  # TODO: find something useful to return
        else:
            result = self.optimizer.optimize(model, dataset)

        self.update_posterior_cache()
        return result

    def get_inducing_variables(self) -> Tuple[TensorType, TensorType, TensorType, bool]:
        """
        Return the model's inducing variables. Note that GPflow's VGP model is
        hard-coded to use the whitened representation.

        :return: Tensors containing: the inducing points (i.e. locations of the inducing
            variables); the variational mean q_mu; the Cholesky decomposition of the
            variational covariance q_sqrt; and a bool denoting if we are using whitened
            or non-whitened representations.
        """
        inducing_points = self.model.data[0]
        q_mu = self.model.q_mu
        q_sqrt = self.model.q_sqrt
        whiten = True  # GPflow's VGP model is hard-coded to use the whitened representation
        return inducing_points, q_mu, q_sqrt, whiten

    def trajectory_sampler(self) -> TrajectorySampler[VariationalGaussianProcess]:
        """
        Return a trajectory sampler. For :class:`VariationalGaussianProcess`, we build
        trajectories using a decoupled random Fourier feature approximation.

        At the moment only models with single latent GP are supported.

        :return: The trajectory sampler.
        :raise NotImplementedError: If we try to use the
            sampler with a model that has more than one latent GP.
        """
        if self.model.num_latent_gps > 1:
            raise NotImplementedError(
                f"""
                Trajectory sampler does not currently support models with multiple latent
                GPs, however received a model with {self.model.num_latent_gps} latent GPs.
                """
            )

        return DecoupledTrajectorySampler(self, self._num_rff_features)

    def covariance_between_points_encoded(
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

        _, _, q_sqrt, whiten = self.get_inducing_variables()

        return _covariance_between_points_for_variational_models(
            kernel=self.get_kernel(),
            inducing_points=self.model.data[0],
            q_sqrt=q_sqrt,
            query_points_1=query_points_1,
            query_points_2=query_points_2,
            whiten=whiten,
        )


class MultifidelityAutoregressive(
    TrainableProbabilisticModel, SupportsPredictY, SupportsCovarianceWithTopFidelity
):
    r"""
    A :class:`TrainableProbabilisticModel` implementation of the model
    from :cite:`Kennedy2000`. This is a multi-fidelity model that works with an
    arbitrary number of fidelities. It relies on there being a linear relationship
    between fidelities, and may not perform well for more complex relationships.
    Precisely, it models the relationship between sequential fidelities as

    .. math:: f_{i}(x) = \rho f_{i-1}(x) + \delta(x)

    where :math:`\rho` is a scalar and :math:`\delta` models the residual between the fidelities.
    The only base models supported in this implementation are :class:`~gpflow.models.GPR` models.
    Note: Currently only supports single output problems.
    """

    def __init__(
        self,
        fidelity_models: Sequence[GaussianProcessRegression],
    ):
        """
        :param fidelity_models: List of
            :class:`~trieste.models.gpflow.models.GaussianProcessRegression`
            models, one for each fidelity. The model at index 0 will be used as the signal model
            for the lowest fidelity and models at higher indices will be used as the residual
            model for each higher fidelity.
        """
        self._num_fidelities = len(fidelity_models)

        self.lowest_fidelity_signal_model = fidelity_models[0]
        # Note: The 0th index in the below is not a residual model, and should not be used.
        self.fidelity_residual_models: Sequence[GaussianProcessRegression] = fidelity_models
        # set this as a Parameter so that we can optimize it
        rho = [
            gpflow.Parameter(1.0, trainable=True, name=f"rho_{i}")
            for i in range(self.num_fidelities - 1)
        ]
        self.rho: list[gpflow.Parameter] = [
            gpflow.Parameter(1.0, trainable=False, name="dummy_variable"),
            *rho,
        ]

    @property
    def num_fidelities(self) -> int:
        return self._num_fidelities

    @inherit_check_shapes
    def predict(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        """
        Predict the marginal mean and variance at query_points.

        :param query_points: Query points with shape [N, D+1], where the
            final column of the final dimension contains the fidelity of the query point
        :return: mean: The mean at query_points with shape [N, P],
                 and var: The variance at query_points with shape [N, P]
        """
        (
            query_points_wo_fidelity,  # [..., N, D]
            query_points_fidelity_col,  # [..., N, 1]
        ) = check_and_extract_fidelity_query_points(
            query_points, max_fidelity=self.num_fidelities - 1
        )

        signal_mean, signal_var = self.lowest_fidelity_signal_model.predict(
            query_points_wo_fidelity
        )  # [..., N, P], [..., N, P]

        for fidelity, (fidelity_residual_model, rho) in enumerate(
            zip(self.fidelity_residual_models, self.rho)
        ):
            if fidelity == 0:
                continue

            # Find indices of query points that need predicting for
            fidelity_float = tf.cast(fidelity, query_points.dtype)
            mask = query_points_fidelity_col >= fidelity_float  # [..., N, 1]
            fidelity_indices = tf.where(mask)[..., :-1]
            # Gather necessary query points and  predict
            fidelity_filtered_query_points = tf.gather_nd(
                query_points_wo_fidelity, fidelity_indices
            )
            (
                filtered_fidelity_residual_mean,
                filtered_fidelity_residual_var,
            ) = fidelity_residual_model.predict(fidelity_filtered_query_points)

            # Scatter predictions back into correct location
            fidelity_residual_mean = tf.tensor_scatter_nd_update(
                signal_mean, fidelity_indices, filtered_fidelity_residual_mean
            )
            fidelity_residual_var = tf.tensor_scatter_nd_update(
                signal_var, fidelity_indices, filtered_fidelity_residual_var
            )

            # Calculate mean and var for all columns (will be incorrect for qps with fid < fidelity)
            new_fidelity_signal_mean = rho * signal_mean + fidelity_residual_mean
            new_fidelity_signal_var = fidelity_residual_var + (rho**2) * signal_var

            # Mask out incorrect values and update mean and var for correct ones
            mask = query_points_fidelity_col >= fidelity_float
            signal_mean = tf.where(mask, new_fidelity_signal_mean, signal_mean)
            signal_var = tf.where(mask, new_fidelity_signal_var, signal_var)

        return signal_mean, signal_var

    def _calculate_residual(self, dataset: Dataset, fidelity: int) -> TensorType:
        r"""
        Calculate the true residuals for a set of datapoints at a given fidelity.

        Dataset should be made up of points that you have observations for at fidelity `fidelity`.
        The residuals calculated here are the difference between the data and the prediction at the
        lower fidelity multiplied by the rho value at this fidelity. This produces the training
        data for the residual models.

        .. math:: r_{i} = y - \rho_{i} * f_{i-1}(x)

        :param dataset: Dataset of points for which to calculate the residuals. Must have
         observations at fidelity `fidelity`. Query points shape is [N, D], observations is [N,P].
        :param fidelity: The fidelity for which to calculate the residuals
        :return: The true residuals at given datapoints for given fidelity, shape is [N,1].
        """
        fidelity_query_points = add_fidelity_column(dataset.query_points, fidelity - 1)
        residuals = (
            dataset.observations - self.rho[fidelity] * self.predict(fidelity_query_points)[0]
        )
        return residuals

    @inherit_check_shapes
    def sample(self, query_points: TensorType, num_samples: int) -> TensorType:
        """
        Sample `num_samples` samples from the posterior distribution at `query_points`

        :param query_points: The query points at which to sample of shape [N, D+1], where the
            final column of the final dimension contains the fidelity of the query point
        :param num_samples: The number of samples (S) to generate for each query point.
        :return: samples from the posterior of shape [..., S, N, P]
        """
        (
            query_points_wo_fidelity,
            query_points_fidelity_col,
        ) = check_and_extract_fidelity_query_points(
            query_points, max_fidelity=self.num_fidelities - 1
        )

        signal_sample = self.lowest_fidelity_signal_model.sample(
            query_points_wo_fidelity, num_samples
        )  # [S, N, P]

        for fidelity in range(1, int(tf.reduce_max(query_points_fidelity_col)) + 1):
            fidelity_residual_sample = self.fidelity_residual_models[fidelity].sample(
                query_points_wo_fidelity, num_samples
            )

            new_fidelity_signal_sample = (
                self.rho[fidelity] * signal_sample + fidelity_residual_sample
            )  # [S, N, P]

            mask = query_points_fidelity_col >= fidelity  # [N, P]
            mask = tf.broadcast_to(mask[..., None, :, :], new_fidelity_signal_sample.shape)

            signal_sample = tf.where(mask, new_fidelity_signal_sample, signal_sample)

        return signal_sample

    @inherit_check_shapes
    def predict_y(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        """
        Predict the marginal mean and variance at `query_points` including observation noise

        :param query_points: Query points with shape [..., N, D+1], where the
            final column of the final dimension contains the fidelity of the query point
        :return: mean: The mean at query_points with shape [N, P],
                 and var: The variance at query_points with shape [N, P]
        """

        f_mean, f_var = self.predict(query_points)
        query_points_fidelity_col = query_points[..., -1:]

        # Get fidelity 0 observation noise
        observation_noise = (
            tf.ones_like(query_points_fidelity_col)
            * self.lowest_fidelity_signal_model.get_observation_noise()
        )

        for fidelity in range(1, self.num_fidelities):
            fidelity_observation_noise = (
                self.rho[fidelity] ** 2
            ) * observation_noise + self.fidelity_residual_models[fidelity].get_observation_noise()

            mask = query_points_fidelity_col >= fidelity

            observation_noise = tf.where(mask, fidelity_observation_noise, observation_noise)

        return f_mean, f_var + observation_noise

    def update(self, dataset: Dataset) -> None:
        """
        Update the models on their corresponding data. The data for each model is
        extracted by splitting the observations in ``dataset`` by fidelity level.

        :param dataset: The query points and observations for *all* the wrapped models.
        """
        check_and_extract_fidelity_query_points(
            dataset.query_points, max_fidelity=self.num_fidelities - 1
        )
        dataset_per_fidelity = split_dataset_by_fidelity(dataset, self.num_fidelities)
        for fidelity, dataset_for_fidelity in enumerate(dataset_per_fidelity):
            if fidelity == 0:
                self.lowest_fidelity_signal_model.update(dataset_for_fidelity)
            else:
                # Make query points but with final column corresponding to
                # fidelity we wish to predict at
                self.fidelity_residual_models[fidelity].update(
                    Dataset(
                        dataset_for_fidelity.query_points,
                        self._calculate_residual(dataset_for_fidelity, fidelity),
                    )
                )

    def optimize(self, dataset: Dataset) -> None:
        """
        Optimize all the models on their corresponding data. The data for each model is
        extracted by splitting the observations in ``dataset``  by fidelity level.
        Note that we have to code up a custom loss function when optimizing our residual
        model, so that we can include the correlation parameter as an optimisation variable.

        :param dataset: The query points and observations for *all* the wrapped models.
        """
        check_and_extract_fidelity_query_points(
            dataset.query_points, max_fidelity=self.num_fidelities - 1
        )
        dataset_per_fidelity = split_dataset_by_fidelity(dataset, self.num_fidelities)

        for fidelity, dataset_for_fidelity in enumerate(dataset_per_fidelity):
            if fidelity == 0:
                self.lowest_fidelity_signal_model.optimize(dataset_for_fidelity)
            else:
                gpf_residual_model = self.fidelity_residual_models[fidelity].model

                fidelity_observations = dataset_for_fidelity.observations
                fidelity_query_points = dataset_for_fidelity.query_points
                prev_fidelity_query_points = add_fidelity_column(
                    dataset_for_fidelity.query_points, fidelity - 1
                )
                predictions_from_lower_fidelity = self.predict(prev_fidelity_query_points)[0]

                def loss() -> TensorType:  # hardcoded log liklihood calculation for residual model
                    residuals = (
                        fidelity_observations - self.rho[fidelity] * predictions_from_lower_fidelity
                    )
                    K = gpf_residual_model.kernel(fidelity_query_points)
                    ks = add_noise_cov(K, gpf_residual_model.likelihood.variance)
                    L = tf.linalg.cholesky(ks)
                    m = gpf_residual_model.mean_function(fidelity_query_points)
                    log_prob = multivariate_normal(residuals, m, L)
                    return -1.0 * tf.reduce_sum(log_prob)

                trainable_variables = (
                    gpf_residual_model.trainable_variables + self.rho[fidelity].variables
                )
                self.fidelity_residual_models[fidelity].optimizer.optimizer.minimize(
                    loss, trainable_variables
                )
                residuals = self._calculate_residual(dataset_for_fidelity, fidelity)
                self.fidelity_residual_models[fidelity].update(
                    Dataset(fidelity_query_points, residuals)
                )

    def covariance_with_top_fidelity(self, query_points: TensorType) -> TensorType:
        """
        Calculate the covariance of the output at `query_point` and a given fidelity with the
        highest fidelity output at the same `query_point`.

        :param query_points: The query points to calculate the covariance for, of shape [N, D+1],
            where the final column of the final dimension contains the fidelity of the query point
        :return: The covariance with the top fidelity for the `query_points`, of shape [N, P]
        """
        fidelities = query_points[..., -1:]  # [..., 1]

        _, f_var = self.predict(query_points)

        for fidelity in range(self.num_fidelities - 1, -1, -1):
            mask = fidelities < fidelity

            f_var = tf.where(mask, f_var, f_var * self.rho[fidelity])

        return f_var

    def log(self, dataset: Optional[Dataset] = None) -> None:
        return


class MultifidelityNonlinearAutoregressive(
    TrainableProbabilisticModel, SupportsPredictY, SupportsCovarianceWithTopFidelity
):
    r"""
    A :class:`TrainableProbabilisticModel` implementation of the model from
    :cite:`perdikaris2017nonlinear`. This is a multifidelity model that works with
    an arbitrary number of fidelities. It is capable of modelling both linear and non-linear
    relationships between fidelities. It models the relationship between sequential fidelities as

    .. math:: f_{i}(x) =  g_{i}(x, f_{*i-1}(x))

    where :math:`f{*i-1}` is the posterior of the previous fidelity.
    The only base models supported in this implementation are :class:`~gpflow.models.GPR` models.
    Note: Currently only supports single output problems.
    """

    def __init__(
        self,
        fidelity_models: Sequence[GaussianProcessRegression],
        num_monte_carlo_samples: int = 100,
    ):
        """
        :param fidelity_models: List of
            :class:`~trieste.models.gpflow.models.GaussianProcessRegression`
            models, one for each fidelity. The model at index 0 should take
            inputs with the same number of dimensions as `x` and can use any kernel,
            whilst the later models should take an extra input dimesion, and use the kernel
            described in :cite:`perdikaris2017nonlinear`.
        :param num_monte_carlo_samples: The number of Monte Carlo samples to use for the
            sections of prediction and sampling that require the use of Monte Carlo methods.
        """

        self._num_fidelities = len(fidelity_models)
        self.fidelity_models = fidelity_models
        self.monte_carlo_random_numbers = tf.random.normal(
            [num_monte_carlo_samples, 1], dtype=tf.float64
        )

    @property
    def num_fidelities(self) -> int:
        return self._num_fidelities

    @inherit_check_shapes
    def sample(self, query_points: TensorType, num_samples: int) -> TensorType:
        """
        Return ``num_samples`` samples from the independent marginal distributions at
        ``query_points``.

        :param query_points: The points at which to sample, with shape [..., N, D].
        :param num_samples: The number of samples at each point.
        :return: The samples, with shape [..., S, N], where S is the number of samples.
        """
        (
            query_points_wo_fidelity,
            query_points_fidelity_col,
        ) = check_and_extract_fidelity_query_points(
            query_points, max_fidelity=self.num_fidelities - 1
        )  # [..., N, D], [..., N, 1]

        signal_sample = self.fidelity_models[0].sample(
            query_points_wo_fidelity, num_samples
        )  # [..., S, N, 1]

        # Repeat query_points to get same shape as signal sample
        query_points_fidelity_col = tf.broadcast_to(
            query_points_fidelity_col[..., None, :, :], signal_sample.shape
        )  # [..., S, N, 1]

        for fidelity in range(1, self.num_fidelities):
            qp_repeated = tf.broadcast_to(
                query_points_wo_fidelity[..., None, :, :],
                signal_sample.shape[:-1] + query_points_wo_fidelity.shape[-1],
            )  # [..., S, N, D]
            qp_augmented = tf.concat([qp_repeated, signal_sample], axis=-1)  # [..., S, N, D + 1]
            new_signal_sample = self.fidelity_models[fidelity].sample(
                qp_augmented, 1
            )  # [..., S, 1, N, 1]
            # Remove second dimension caused by getting a single sample
            new_signal_sample = new_signal_sample[..., :, 0, :, :]

            mask = query_points_fidelity_col >= fidelity

            signal_sample = tf.where(mask, new_signal_sample, signal_sample)

        return signal_sample

    @inherit_check_shapes
    def predict(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        """
        Predict the marginal mean and variance at query_points.

        :param query_points: Query points with shape [..., N, D+1], where the
            final column of the final dimension contains the fidelity of the query point
        :return: mean: The mean at query_points with shape [..., N, P],
            and var: The variance at query_points with shape [..., N, P]
        """
        check_and_extract_fidelity_query_points(query_points, max_fidelity=self.num_fidelities - 1)

        sample_mean, sample_var = self._sample_mean_and_var_at_fidelities(
            query_points
        )  # [..., N, 1, S], [..., N, 1, S]
        variance = tf.reduce_mean(sample_var, axis=-1) + tf.math.reduce_variance(
            sample_mean, axis=-1
        )
        mean = tf.reduce_mean(sample_mean, axis=-1)
        return mean, variance

    @inherit_check_shapes
    def predict_y(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        """
        Predict the marginal mean and variance at `query_points` including observation noise

        :param query_points: Query points with shape [..., N, D+1], where the
            final column of the final dimension contains the fidelity of the query point
        :return: mean: The mean at query_points with shape [N, P],
            and var: The variance at query_points with shape [N, P]
        """
        _, query_points_fidelity_col = check_and_extract_fidelity_query_points(
            query_points, max_fidelity=self.num_fidelities - 1
        )

        f_mean, f_var = self.predict(query_points)

        # Get fidelity 0 observation noise
        observation_noise = (
            tf.ones_like(query_points_fidelity_col)
            * self.fidelity_models[0].get_observation_noise()
        )

        for fidelity in range(1, self.num_fidelities):
            fidelity_observation_noise = self.fidelity_models[fidelity].get_observation_noise()

            mask = query_points_fidelity_col >= fidelity

            observation_noise = tf.where(mask, fidelity_observation_noise, observation_noise)
        return f_mean, f_var + observation_noise

    def _sample_mean_and_var_at_fidelities(
        self, query_points: TensorType
    ) -> tuple[TensorType, TensorType]:
        """
        Draw `num_monte_carlo_samples` samples of mean and variance from the model at the fidelities
        passed in the final column of the query points.

        :param query_points:  Query points with shape [..., N, D+1], where the
            final column of the final dimension contains the fidelity of the query point
        :return: sample_mean: Samples of the mean at the query points with shape [..., N, 1, S]
            and sample_var: Samples of the variance at the query points with shape [..., N, 1, S]
        """

        (
            query_points_wo_fidelity,
            query_points_fidelity_col,
        ) = check_and_extract_fidelity_query_points(
            query_points, max_fidelity=self.num_fidelities - 1
        )  # [..., N, D], [..., N, 1]

        sample_mean, sample_var = self.fidelity_models[0].predict(
            query_points_wo_fidelity
        )  # [..., N, 1], [..., N, 1]

        # Create new dimension to store samples for each query point
        # Repeat the inital sample mean and variance S times and add a dimension in the
        # middle (so that sample mean and query points can be concatenated sensibly)
        sample_mean = tf.broadcast_to(
            sample_mean, sample_mean.shape[:-1] + self.monte_carlo_random_numbers.shape[0]
        )[
            ..., :, None, :
        ]  # [..., N, 1, S]
        sample_var = tf.broadcast_to(
            sample_var, sample_var.shape[:-1] + self.monte_carlo_random_numbers.shape[0]
        )[
            ..., :, None, :
        ]  # [..., N, 1, S]

        # Repeat fidelity points for each sample to match shapes for masking
        query_points_fidelity_col = tf.broadcast_to(
            query_points_fidelity_col,
            query_points_fidelity_col.shape[:-1] + self.monte_carlo_random_numbers.shape[0],
        )[
            ..., :, None, :
        ]  # [..., N, 1, S]

        # Predict for all fidelities but stop updating once we have
        # reached desired fidelity for each query point
        for fidelity in range(1, self.num_fidelities):
            # sample_mean [..., N, 1, S]
            # sample_var [..., N, 1, S]
            (
                next_fidelity_sample_mean,
                next_fidelity_sample_var,
            ) = self._propagate_samples_through_level(
                query_points_wo_fidelity, fidelity, sample_mean, sample_var
            )

            mask = query_points_fidelity_col >= fidelity  # [..., N, 1, S]

            sample_mean = tf.where(mask, next_fidelity_sample_mean, sample_mean)  # [..., N, 1, S]
            sample_var = tf.where(mask, next_fidelity_sample_var, sample_var)  # [..., N, 1, S]

        return sample_mean, sample_var

    def _propagate_samples_through_level(
        self,
        query_point: TensorType,
        fidelity: int,
        sample_mean: TensorType,
        sample_var: TensorType,
    ) -> tuple[TensorType, TensorType]:
        """
        Propagate samples through a given fidelity.

        This takes a set of query points without a fidelity column and calculates samples
        at the given fidelity, using the sample means and variances from the previous fidelity.

        :param query_points: The query points to sample at, with no fidelity column,
            with shape[..., N, D]
        :param fidelity: The fidelity to propagate the samples through
        :param sample_mean: Samples of the posterior mean at the previous fidelity,
            with shape [..., N, 1, S]
        :param sample_var: Samples of the posterior variance at the previous fidelity,
            with shape [..., N, 1, S]
        :return: sample_mean: Samples of the posterior mean at the given fidelity,
            of shape [..., N, 1, S]
            and sample_var: Samples of the posterior variance at the given fidelity,
            of shape [..., N, 1, S]
        """
        # Repeat random numbers for each query point and add middle dimension
        # for concatentation with query points This also means that it has the same
        # shape as sample_var and sample_mean, so there's no broadcasting required
        # Note: at the moment we use the same monte carlo values for every value in the batch dim

        reshaped_random_numbers = tf.broadcast_to(
            tf.transpose(self.monte_carlo_random_numbers)[..., None, :],
            sample_mean.shape,
        )  # [..., N, 1, S]
        samples = reshaped_random_numbers * tf.sqrt(sample_var) + sample_mean  # [..., N, 1, S]
        # Add an extra unit dim to query_point and repeat for each of the samples
        qp_repeated = tf.broadcast_to(
            query_point[..., :, :, None],  # [..., N, D, 1]
            query_point.shape + samples.shape[-1],
        )  # [..., N, D, S]
        qp_augmented = tf.concat([qp_repeated, samples], axis=-2)  # [..., N, D+1, S]

        # Flatten sample dimension to n_qp dimension to pass through predictor
        # Switch dims to make reshape match up correct dimensions for query points
        # Use Einsum to switch last two dimensions
        qp_augmented = tf.linalg.matrix_transpose(qp_augmented)  # [..., N, S, D+1]

        flat_qp_augmented, unflatten = flatten_leading_dims(qp_augmented)  # [...*N*S, D+1]

        # Dim of flat qp augmented is now [n_qps*n_samples, qp_dims], as the model expects
        sample_mean, sample_var = self.fidelity_models[fidelity].predict(
            flat_qp_augmented
        )  # [...*N*S, 1], [...*N*S, 1]

        # Reshape back to have samples as final dimension
        sample_mean = unflatten(sample_mean)  # [..., N, S, 1]
        sample_var = unflatten(sample_var)  # [..., N, S, 1]
        sample_mean = tf.linalg.matrix_transpose(sample_mean)  # [..., N, 1, S]
        sample_var = tf.linalg.matrix_transpose(sample_var)  # [..., N, 1, S]

        return sample_mean, sample_var

    def update(self, dataset: Dataset) -> None:
        """
        Update the models on their corresponding data. The data for each model is
        extracted by splitting the observations in ``dataset`` by fidelity level.

        :param dataset: The query points and observations for *all* the wrapped models.
        """
        check_and_extract_fidelity_query_points(
            dataset.query_points, max_fidelity=self.num_fidelities - 1
        )
        dataset_per_fidelity = split_dataset_by_fidelity(
            dataset, num_fidelities=self.num_fidelities
        )
        for fidelity, dataset_for_fidelity in enumerate(dataset_per_fidelity):
            if fidelity == 0:
                self.fidelity_models[0].update(dataset_for_fidelity)
            else:
                cur_fidelity_model = self.fidelity_models[fidelity]
                new_final_query_point_col, _ = self.predict(
                    add_fidelity_column(dataset_for_fidelity.query_points, fidelity - 1)
                )
                new_query_points = tf.concat(
                    [dataset_for_fidelity.query_points, new_final_query_point_col], axis=1
                )
                cur_fidelity_model.update(
                    Dataset(new_query_points, dataset_for_fidelity.observations)
                )

    def optimize(self, dataset: Dataset) -> None:
        """
        Optimize all the models on their corresponding data. The data for each model is
        extracted by splitting the observations in ``dataset``  by fidelity level.

        :param dataset: The query points and observations for *all* the wrapped models.
        """
        check_and_extract_fidelity_query_points(
            dataset.query_points, max_fidelity=self.num_fidelities - 1
        )
        dataset_per_fidelity = split_dataset_by_fidelity(dataset, self.num_fidelities)

        for fidelity, dataset_for_fidelity in enumerate(dataset_per_fidelity):
            if fidelity == 0:
                self.fidelity_models[0].optimize(dataset_for_fidelity)
            else:
                fidelity_observations = dataset_for_fidelity.observations
                fidelity_query_points = dataset_for_fidelity.query_points
                prev_fidelity_query_points = add_fidelity_column(
                    fidelity_query_points, fidelity - 1
                )
                means_from_lower_fidelity = self.predict(prev_fidelity_query_points)[0]
                augmented_qps = tf.concat(
                    [fidelity_query_points, means_from_lower_fidelity], axis=1
                )
                self.fidelity_models[fidelity].optimize(
                    Dataset(augmented_qps, fidelity_observations)
                )
                self.fidelity_models[fidelity].update(Dataset(augmented_qps, fidelity_observations))

    def covariance_with_top_fidelity(self, query_points: TensorType) -> TensorType:
        """
        Calculate the covariance of the output at `query_point` and a given fidelity with the
        highest fidelity output at the same `query_point`.

        :param query_points: The query points to calculate the covariance for, of shape [N, D+1],
            where the final column of the final dimension contains the fidelity of the query point
        :return: The covariance with the top fidelity for the `query_points`, of shape [N, P]
        """
        num_samples = 100
        (
            query_points_wo_fidelity,
            query_points_fidelity_col,
        ) = check_and_extract_fidelity_query_points(
            query_points, max_fidelity=self.num_fidelities - 1
        )  # [N, D], [N, 1]

        # Signal sample stops updating once fidelity is reached for that query point
        signal_sample = self.fidelity_models[0].model.predict_f_samples(
            query_points_wo_fidelity, num_samples, full_cov=False
        )

        # Repeat query_points to get same shape as signal sample
        query_points_fidelity_col = tf.broadcast_to(
            query_points_fidelity_col[None, :, :], signal_sample.shape
        )
        # Max fidelity sample keeps updating to the max fidelity
        max_fidelity_sample = tf.identity(signal_sample)

        for fidelity in range(1, self.num_fidelities):
            qp_repeated = tf.broadcast_to(
                query_points_wo_fidelity[None, :, :],
                tf.TensorShape(num_samples) + query_points_wo_fidelity.shape,
            )  # [S, N, D]
            # We use max fidelity sample here, which is okay because anything
            # with a lower fidelity will not be updated
            qp_augmented = tf.concat([qp_repeated, max_fidelity_sample], axis=-1)  # [S, N, D + 1]
            new_signal_sample = self.fidelity_models[fidelity].model.predict_f_samples(
                qp_augmented, 1, full_cov=False
            )
            # Remove second dimension caused by getting a single sample
            new_signal_sample = new_signal_sample[:, 0, :, :]

            mask = query_points_fidelity_col >= fidelity

            signal_sample = tf.where(mask, new_signal_sample, signal_sample)  # [S, N, 1]
            max_fidelity_sample = new_signal_sample  # [S, N , 1]

        cov = tfp.stats.covariance(signal_sample, max_fidelity_sample)[:, :, 0]

        return cov

    def log(self, dataset: Optional[Dataset] = None) -> None:
        return
