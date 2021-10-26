# %% [markdown]
# # S-GP-TS Demo

# %% [markdown]
# This code accomapnies the paper "Scalable Thompson Sampling usingSparse Gaussian Process Models".
#
# First we demonstrate the method on a simple 2D benchmark, before showing how more complicated experiments can be ran.

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from __future__ import annotations

import numpy as np
import tensorflow as tf
import trieste
from trieste.objectives import scaled_branin
from trieste.objectives.utils import mk_observer
from trieste.utils.inducing_point_selectors import KMeans
import tensorflow_probability as tfp
import gpflow
import gpflux
from gpflow.config import default_float
from gpflux.layers.basis_functions.random_fourier_features import RandomFourierFeatures
from gpflux.sampling.kernel_with_feature_decomposition import KernelWithFeatureDecomposition
from gpflux.helpers import construct_basic_inducing_variables
from tensorflow_probability.python.distributions.laplace import Laplace
from trieste.acquisition.rule import OBJECTIVE
from trieste.acquisition.function import SingleModelGreedyAcquisitionBuilder, AcquisitionFunction
from gpflow.inducing_variables import InducingPoints
from gpflux.layers import GPLayer, LatentVariableLayer
from gpflux.models import DeepGP
from gpflux.models.deep_gp import sample_dgp
from trieste.data import Dataset
from trieste.models.gpflux.models import DeepGaussianProcess
from trieste.models.gpflux.utils import InducingPointSelector, KMeans
from typing import Callable, Dict, Any, Optional
from trieste.types import TensorType
import matplotlib.pyplot as plt
from util.plotting import plot_regret, create_grid, plot_gp_2d, plot_function_2d, plot_bo_points
from gpflow.config import default_float
from gpflow.inducing_variables import InducingVariables, SharedIndependentInducingVariables
from gpflow.kernels import SeparateIndependent

from gpflux.sampling.kernel_with_feature_decomposition import KernelWithFeatureDecomposition
from gpflux.sampling.sample import efficient_sample, Sample
from gpflux.sampling.utils import draw_conditional_sample

np.random.seed(1789)
tf.random.set_seed(1789)
tf.keras.backend.set_floatx("float64")

# %% [markdown]
# ## Describe the problem
# %%
num_initial_points = 100
num_inducing_points = 50
num_rff = 1000
num_bo_iterations = 10
batch_size = 20

search_space = trieste.space.Box([0, 0], [1, 1])
noise = .5
inducing_point_selector = KMeans(search_space)
use_logn_noise = True
quantile_level = 0.9
beta = tfp.distributions.Normal(loc=0., scale=1.).quantile(value=0.9).numpy()

if use_logn_noise:
    def noisy_branin(x):
        y = scaled_branin(x)
        return y + tf.exp(tf.random.normal(y.shape, stddev=noise * tf.reduce_sum(x, axis=-1, keepdims=True), dtype=y.dtype))
    def quantile_branin(x):
        y = scaled_branin(x)
        return y + tf.exp(beta * noise * tf.reduce_sum(x, axis=-1, keepdims=True))
else:
    def noisy_branin(x):
        y = scaled_branin(x)
        return y + (tf.random.normal(y.shape, stddev=noise * tf.reduce_sum(x, axis=-1, keepdims=True), dtype=y.dtype))
    def quantile_branin(x):
        y = scaled_branin(x)
        return y + beta * noise * tf.reduce_sum(x, axis=-1, keepdims=True)


observer = mk_observer(noisy_branin, OBJECTIVE)

initial_query_points = search_space.sample(num_initial_points)
initial_data = observer(initial_query_points)
num_data, input_dim = initial_data[OBJECTIVE].query_points.shape


@efficient_sample.register(
    SharedIndependentInducingVariables,
    SeparateIndependent,
    object
)
def _efficient_sample_matheron_rule(
    inducing_variable: InducingVariables,
    kernel: KernelWithFeatureDecomposition,
    q_mu: tf.Tensor,
    *,
    q_sqrt: Optional[TensorType] = None,
    whiten: bool = False,
) -> Sample:
    samples = []
    for i, k in enumerate(kernel.kernels):
        samples.append(efficient_sample(inducing_variable.inducing_variable, k, q_mu[..., i:(i+1)],
                                        q_sqrt=q_sqrt[i:(i+1), ...], whiten=whiten))

    class MultiOutputSample(Sample):
        def __call__(self, X: TensorType) -> tf.Tensor:
            return tf.concat([s(X) for s in samples], axis=-1)
    return MultiOutputSample()


class ASymmetricLaplace(Laplace):
    def __init__(self,
                 loc,
                 scale,
                 validate_args=False,
                 allow_nan_stats=True,
                 name='Laplace',
                 tau=quantile_level):

        super().__init__(loc, scale, validate_args, allow_nan_stats, name)

        # with tf.name_scope(name) as name:
        # dtype = dtype_util.common_dtype([loc, scale], tf.float32)
        # self.tau = tensor_util.convert_nonref_to_tensor(
        #     tau, name='tau', dtype=dtype)

        self.tau = tau

    # @property
    # def tau(self):
    #     return self._tau

    def _mean(self):
        loc = tf.convert_to_tensor(self.loc)
        return tf.broadcast_to(loc + self.scale * (1. - 2 * self.tau) / (self.tau * (1. - self.tau)),
                               self._batch_shape_tensor(loc=loc))

    def _stddev(self):
        scale = tf.convert_to_tensor(self.scale)
        return tf.broadcast_to(scale * np.sqrt(1. - 2. * self.tau + 2. * self.tau**2) /
                                     (self.tau * (1. - self.tau)),
                               self._batch_shape_tensor(scale=scale))

    def _variance(self):
        scale = tf.convert_to_tensor(self.scale)
        return tf.broadcast_to(scale**2 * (1. - 2. * self.tau + 2. * self.tau**2) /
                                     (self.tau**2 * (1. - self.tau)**2),
                               self._batch_shape_tensor(scale=scale))

    def _log_prob(self, x):
        loc = tf.convert_to_tensor(self.loc)
        scale = tf.convert_to_tensor(self.scale)
        z = (x - loc) / scale
        is_neg = 0.5 - 0.5 * tf.sign(z)
        return tf.math.log(self.tau * (1 - self.tau) / scale) - z * (self.tau - is_neg)

    def _cdf(self, x):
        z = self._z(x)
        is_neg = 0.5 - 0.5 * tf.sign(z)
        negF = self.tau * tf.exp((1. - self.tau) / self.scale * (x - self.loc))
        posF = 1. - (1 - self.tau) + tf.exp(-self.tau/self.scale * (x - self.loc))
        return tf.where(is_neg > 0.5, negF, posF)

    def _log_cdf(self, x):
        return tf.math.log(self._cdf(x))

    def _quantile(self, p):
        loc = tf.convert_to_tensor(self.loc)
        scale = tf.convert_to_tensor(self.scale)
        q1 = loc + scale / (1. - self.tau) * tf.math.log(p / self.tau)
        q2 = loc - scale / self.tau * tf.math.log((1. - p) / (1. - self.tau))
        return tf.where(p > self.tau, q1, q2)

    def _z(self, x):
        return (x - self.loc) / self.scale

    def _sample_n(self, n, seed=None):
        return NotImplementedError

    def _log_survival_function(self, x):
        return NotImplementedError

    def _entropy(self):
        return NotImplementedError

    def _median(self):
        return NotImplementedError

    def _mode(self):
        return NotImplementedError


class FeaturedHetGPFluxModel(DeepGaussianProcess):

    def __init__(self,
                 model: DeepGP,
                 optimizer: tf.optimizers.Optimizer | None = None,
                 fit_args: Dict[str, Any] | None = None,
                 inducing_point_selector: InducingPointSelector = None,
                 ):

        super().__init__(model, optimizer, fit_args)

        if inducing_point_selector is None:
            inducing_point_selector = KMeans
        self._inducing_point_selector = inducing_point_selector

    def sample_trajectory(self) -> Callable:
        return sample_dgp(self.model_gpflux)

    def update(self, dataset: Dataset) -> None:
        inputs = dataset.query_points
        new_num_data = inputs.shape[0]
        self.model_gpflux.num_data = new_num_data

        # Update num_data for each layer, as well as make sure dataset shapes are ok
        for i, layer in enumerate(self.model_gpflux.f_layers):
            if hasattr(layer, "num_data"):
                layer.num_data = new_num_data

            if isinstance(layer, LatentVariableLayer):
                inputs = layer(inputs)
                continue

            if isinstance(layer.inducing_variable, InducingPoints):
                inducing_variable = layer.inducing_variable
            else:
                inducing_variable = layer.inducing_variable.inducing_variable

            if inputs.shape[-1] != inducing_variable.Z.shape[-1]:
                raise ValueError(
                    f"Shape {inputs.shape} of input to layer {layer} is incompatible with shape"
                    f" {inducing_variable.Z.shape} of that layer. Trailing dimensions must match."
                )
            inputs = layer(inputs)

            if hasattr(layer.kernel, 'feature_functions'): # If using RFF kernel decomp then need to resample for new kernel params
                feature_function = layer.kernel.feature_functions
                input_shape = dataset.query_points.shape
                def renew_rff(feature_f, input_dim):
                    shape_bias = [1, feature_f.output_dim]
                    new_b = feature_f._sample_bias(shape_bias, dtype=feature_f.dtype)
                    feature_f.b = new_b
                    shape_weights = [feature_f.output_dim, input_dim]
                    new_W = feature_f._sample_weights(shape_weights, dtype=feature_f.dtype)
                    feature_f.W = new_W
                renew_rff(feature_function,  input_shape[-1])

            num_inducing = layer.inducing_variable.inducing_variable.Z.shape[0]

            Z = self._inducing_point_selector.get_points(dataset.query_points,
                                                         dataset.observations,
                                                         num_inducing,
                                                         layer.kernel,
                                                         noise=1e-6)

            jitter = 1e-6

            if layer.whiten:
                f_mu, f_cov = self.predict_joint(Z)  # [N, L], [L, N, N]
                Knn = layer.kernel(Z, full_cov=True)  # [N, N]
                jitter_mat = jitter * tf.eye(num_inducing, dtype=Knn.dtype)
                Lnn = tf.linalg.cholesky(Knn + jitter_mat)  # [N, N]
                new_q_mu = tf.linalg.triangular_solve(Lnn, f_mu)  # [N, L]
                tmp = tf.linalg.triangular_solve(Lnn[None], f_cov)  # [L, N, N], L⁻¹ f_cov
                S_v = tf.linalg.triangular_solve(Lnn[None], tf.linalg.matrix_transpose(tmp))  # [L, N, N]
                new_q_sqrt = tf.linalg.cholesky(S_v + jitter_mat)  # [L, N, N]
            else:
                new_q_mu, new_f_cov = layer.predict(Z, full_cov=True)  # [N, L], [L, N, N]
                jitter_mat = jitter * tf.eye(num_inducing, dtype=new_f_cov.dtype)
                new_q_sqrt = tf.linalg.cholesky(new_f_cov + jitter_mat)

            layer.q_mu.assign(new_q_mu)
            layer.q_sqrt.assign(new_q_sqrt)
            layer.inducing_variable.inducing_variable.Z.assign(Z)


class NegativeGaussianProcessTrajectory(SingleModelGreedyAcquisitionBuilder):
    def __repr__(self) -> str:
        return f"NegativeGaussianProcessTrajectory"

    def prepare_acquisition_function(
        self, dataset: Dataset, model: FeaturedHetGPFluxModel,
        pending_points: Optional[TensorType] = None,
    ) -> AcquisitionFunction:
        trajectory = model.sample_trajectory()
        return lambda at: -trajectory(tf.squeeze(at, axis=1))[..., 0:1]


class NegativeQuantilefromGaussianHetGPTrajectory(SingleModelGreedyAcquisitionBuilder):

    def __init__(self, quantile_level: float = 0.9):
        self._quantile_level = quantile_level

    def __repr__(self) -> str:
        return f"NegativeGaussianProcessTrajectory"

    def prepare_acquisition_function(
        self, dataset: Dataset, model: FeaturedHetGPFluxModel,
        pending_points: Optional[TensorType] = None,
    ) -> AcquisitionFunction:
        trajectory = model.sample_trajectory()

        def quantile_traj(at):
            lik_layer = model.model_gpflux.likelihood_layer
            dist = lik_layer.likelihood.conditional_distribution(trajectory(tf.squeeze(at, axis=1)))
            return -dist.quantile(value=self._quantile_level)
        return quantile_traj


def create_kernel_with_features(var, input_dim, num_rff):
    kernel = gpflow.kernels.Matern52(variance=var, lengthscales=0.2 * np.ones(input_dim, ))
    prior_scale = tf.cast(1.0, dtype=tf.float64)
    kernel.variance.prior = tfp.distributions.LogNormal(tf.cast(0.0, dtype=tf.float64), prior_scale)
    kernel.lengthscales.prior = tfp.distributions.LogNormal(tf.math.log(kernel.lengthscales), prior_scale)

    coefficients = np.ones((num_rff, 1), dtype=default_float())
    features = RandomFourierFeatures(kernel, num_rff, dtype=default_float())
    return KernelWithFeatureDecomposition(kernel, features, coefficients)

def build_hetgp_rff_model(data, num_rff=1000, likelihood_distribution=ASymmetricLaplace):
    var = tf.math.reduce_variance(data.observations)
    kernel_with_features1 =create_kernel_with_features(var, input_dim, num_rff)
    kernel_with_features2 = create_kernel_with_features(var / 2., input_dim, num_rff)
    kernel_list = [kernel_with_features1, kernel_with_features2]
    kernel = gpflux.helpers.construct_basic_kernel(kernel_list)

    Z = inducing_point_selector.get_points(data.query_points, data.observations,
                                           num_inducing_points, kernel, noise)
    inducing_variable = construct_basic_inducing_variables(num_inducing_points, input_dim,
                                                           output_dim=2, share_variables=True, z_init= Z)
    gpflow.utilities.set_trainable(inducing_variable, False)

    layer = gpflux.layers.GPLayer(kernel, inducing_variable, num_data, whiten=False,
                                  num_latent_gps=2, mean_function=gpflow.mean_functions.Constant())

    likelihood = gpflow.likelihoods.HeteroskedasticTFPConditional(
        distribution_class=likelihood_distribution,
        scale_transform=tfp.bijectors.Exp(),
    )

    likelihood_layer = gpflux.layers.LikelihoodLayer(likelihood)
    model = gpflux.models.DeepGP([layer], likelihood_layer)

    epochs = 100
    batch_size = 200
    optimizer = tf.optimizers.Adam(0.05)

    callbacks = [tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", patience=10, factor=0.5, verbose=1, min_lr=1e-6),
        tf.keras.callbacks.EarlyStopping(monitor="loss", patience=50, min_delta=0.01, verbose=1, mode="min"),]

    fit_args = {
        "batch_size": batch_size,
        "epochs": epochs,
        "verbose": 2,
        "callbacks": callbacks,
    }

    return FeaturedHetGPFluxModel(model=model, optimizer=optimizer, fit_args=fit_args,
                                  inducing_point_selector=inducing_point_selector)


quantile_model = build_hetgp_rff_model(initial_data[OBJECTIVE], num_rff)
quantile_model.optimize(initial_data[OBJECTIVE])
quantile_models = {OBJECTIVE: quantile_model}

fig, _ = plot_gp_2d(quantile_model.model_gpflux, search_space.lower, search_space.upper, grid_density=30)
fig.axes[0].scatter(initial_query_points[:, 0], initial_query_points[:, 1],
                    initial_data[OBJECTIVE].observations.numpy())


hetgp_model = build_hetgp_rff_model(initial_data[OBJECTIVE], num_rff,
                                    likelihood_distribution=tfp.distributions.Normal)
hetgp_model.optimize(initial_data[OBJECTIVE])
hetgp_models = {OBJECTIVE: hetgp_model}

fig, _ = plot_gp_2d(hetgp_model.model_gpflux, search_space.lower, search_space.upper, grid_density=30)
fig.axes[0].scatter(initial_query_points[:, 0], initial_query_points[:, 1],
                    initial_data[OBJECTIVE].observations.numpy())


# %% [markdown]
# ## Run the optimization loop

# %% [markdown]

# %%
quantile_traj = NegativeGaussianProcessTrajectory()
quantile_rule = trieste.acquisition.rule.EfficientGlobalOptimization(quantile_traj.using(OBJECTIVE),
                                                                     num_query_points=batch_size)
quantile_bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
quantile_result = quantile_bo.optimize(num_bo_iterations, initial_data, quantile_models,
                                       acquisition_rule=quantile_rule, track_state=False)

hetgp_traj = NegativeQuantilefromGaussianHetGPTrajectory(quantile_level=quantile_level)
hetgp_rule = trieste.acquisition.rule.EfficientGlobalOptimization(hetgp_traj.using(OBJECTIVE),
                                                                     num_query_points=batch_size)
hetgp_bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
hetgp_result = hetgp_bo.optimize(num_bo_iterations, initial_data, hetgp_models,
                                       acquisition_rule=hetgp_rule, track_state=False)

# %% [markdown]
# ## Explore the results

# %%
grid_quantile = quantile_branin(create_grid(search_space.lower, search_space.upper, grid_density=200)[0])
actual_minimum = tf.reduce_min(grid_quantile).numpy()

def plot_results(result):
    final_dataset = result.try_get_final_datasets()[OBJECTIVE]
    final_model = result.try_get_final_model().model_gpflux

    query_points = final_dataset.query_points.numpy()
    observations = final_dataset.observations.numpy()
    true_scores = quantile_branin(final_dataset.query_points).numpy()
    arg_min_idx = tf.squeeze(tf.argmin(true_scores, axis=0))

    _, ax = plot_function_2d(
        quantile_branin, search_space.lower, search_space.upper, grid_density=30, contour=True
    )
    plot_bo_points(query_points, ax[0, 0], num_initial_points, arg_min_idx)

    fig, ax = plt.subplots(1, 2)
    plot_regret(true_scores - actual_minimum, ax[0], num_init=num_initial_points, idx_best=arg_min_idx)
    ax[0].set_ylim(0.00001, 1000)
    ax[0].set_yscale("log")
    plot_bo_points(query_points, ax[1], num_init=num_initial_points, idx_best=arg_min_idx)
    fig.show()

    fig, _ = plot_gp_2d(final_model, search_space.lower, search_space.upper, grid_density=30)
    fig.axes[0].scatter(query_points[:, 0], query_points[:, 1], observations)

plot_results(quantile_result)
plot_results(hetgp_result)
