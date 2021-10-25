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
from trieste.models.gpflux.models import FeaturedHetGPFluxModel
from gpflux.helpers import construct_basic_inducing_variables
from tensorflow_probability.python.distributions.laplace import Laplace
from trieste.acquisition.rule import OBJECTIVE


np.random.seed(1789)
tf.random.set_seed(1789)
tf.keras.backend.set_floatx("float64")

# %% [markdown]
# ## Describe the problem
# %%
num_initial_points = 250
num_inducing_points = 50
num_rff = 1000
num_bo_iterations = 1
batch_size = 1

search_space = trieste.space.Box([0, 0], [1, 1])
noise = 1.
inducing_point_selector = KMeans(search_space)

def noisy_branin(x):
    y = scaled_branin(x)
    return y + tf.random.normal(y.shape, stddev=noise * tf.reduce_sum(x, axis=-1, keepdims=True), dtype=y.dtype)

def quantile_branin(x):
    y = scaled_branin(x)
    return y + noise * tf.reduce_sum(x, axis=-1, keepdims=True)



observer = mk_observer(noisy_branin, OBJECTIVE)

initial_query_points = search_space.sample(num_initial_points)
initial_data = observer(initial_query_points)
num_data, input_dim = initial_data[OBJECTIVE].query_points.shape



class ASymmetricLaplace(Laplace):
    def __init__(self,
                 loc,
                 scale,
                 validate_args=False,
                 allow_nan_stats=True,
                 name='Laplace',
                 tau=0.9):

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
        # return is_neg * negF + (1. - is_neg) * posF

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


def create_kernel_with_features(var, input_dim, num_rff):

    kernel = gpflow.kernels.Matern52(variance=var, lengthscales=0.2 * np.ones(input_dim, ))
    prior_scale = tf.cast(1.0, dtype=tf.float64)
    kernel.variance.prior = tfp.distributions.LogNormal(tf.cast(0.0, dtype=tf.float64), prior_scale)
    kernel.lengthscales.prior = tfp.distributions.LogNormal(tf.math.log(kernel.lengthscales), prior_scale)

    coefficients = np.ones((num_rff, 1), dtype=default_float())
    features = RandomFourierFeatures(kernel, num_rff, dtype=default_float())
    return KernelWithFeatureDecomposition(kernel, features, coefficients)

def build_hetgp_rff_model(data, num_rff=1000):
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

    layer = gpflux.layers.GPLayer(
        kernel,
        inducing_variable,
        num_data,
        whiten=False,
        num_latent_gps=2,
        mean_function=gpflow.mean_functions.Constant(),
    )

    likelihood = gpflow.likelihoods.HeteroskedasticTFPConditional(
        distribution_class=ASymmetricLaplace,
        scale_transform=tfp.bijectors.Exp(),
    )

    likelihood_layer = gpflux.layers.LikelihoodLayer(likelihood)
    model = gpflux.models.DeepGP([layer], likelihood_layer)

    epochs = 100
    batch_size = 200
    optimizer = tf.optimizers.Adam(0.05)

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss", patience=10, factor=0.5, verbose=1, min_lr=1e-6,
        ),
        tf.keras.callbacks.EarlyStopping(monitor="loss", patience=50, min_delta=0.01, verbose=1,
                                         mode="min"),
    ]

    fit_args = {
        "batch_size": batch_size,
        "epochs": epochs,
        "verbose": 2,
        "callbacks": callbacks,
    }

    return FeaturedHetGPFluxModel(model=model, optimizer=optimizer, fit_args=fit_args,
                                  inducing_point_selector=inducing_point_selector)


model = build_hetgp_rff_model(initial_data[OBJECTIVE], num_rff)
model.optimize(initial_data[OBJECTIVE])
models = {OBJECTIVE: model}

from util.plotting import plot_gp_2d
fig, _ = plot_gp_2d(model.model_gpflux, search_space.lower,
    search_space.upper,
    grid_density=30)

fig.axes[0].scatter(initial_query_points[:, 0], initial_query_points[:, 1],
                    initial_data[OBJECTIVE].observations.numpy())


# %% [markdown]
# ## Run the optimization loop

# %% [markdown]

# %%
neg_traj = trieste.acquisition.NegativeGaussianProcessTrajectory()
rule = trieste.acquisition.rule.EfficientGlobalOptimization(neg_traj.using(OBJECTIVE), num_query_points=batch_size)

bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
result = bo.optimize(num_bo_iterations, initial_data, models, acquisition_rule=rule, track_state=False)

# %% [markdown]
# ## Explore the results

# %%
dataset = result.try_get_final_datasets()[OBJECTIVE]
query_points = dataset.query_points.numpy()
observations = dataset.observations.numpy()
true_scores = quantile_branin(dataset.query_points).numpy()
arg_min_idx = tf.squeeze(tf.argmin(true_scores, axis=0))

print(f"Believed optima: {query_points[arg_min_idx, :]}")
print(f"Objective function value: {true_scores[arg_min_idx, :]}")

# %% [markdown]

# %%
from util.plotting import plot_function_2d, plot_bo_points

_, ax = plot_function_2d(
    quantile_branin, search_space.lower, search_space.upper, grid_density=30, contour=True
)
plot_bo_points(query_points, ax[0, 0], num_initial_points, arg_min_idx)

# %% [markdown]
# %%
import matplotlib.pyplot as plt
from util.plotting import plot_regret, create_grid

fig, ax = plt.subplots(1, 2)
grid_quantile = quantile_branin(create_grid(search_space.lower, search_space.upper, grid_density=200)[0])
actual_minimum = tf.reduce_min(true_scores).numpy()
plot_regret(true_scores - actual_minimum, ax[0], num_init=num_initial_points, idx_best=arg_min_idx)
ax[0].set_ylim(0.00001, 1000)
ax[0].set_yscale("log")
plot_bo_points(query_points, ax[1], num_init=num_initial_points, idx_best=arg_min_idx)
fig.show()

from util.plotting import plot_gp_2d

fig, _ = plot_gp_2d(result.try_get_final_model().model_gpflux, search_space.lower,
    search_space.upper,
    grid_density=30)

fig.axes[0].scatter(query_points[:, 0], query_points[:, 1], observations)
