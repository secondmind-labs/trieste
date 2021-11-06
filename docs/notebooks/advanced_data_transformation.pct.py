# %% [markdown]
# # Advanced Data Transformation

# %% [markdown]
#
# In advanced cases the user may want to update normalization and model parameters based on incoming data. This can be achieved by overloading the appropriate methods of `DataTransformModelWrapper`.
#
# The example use case here will be when we have a relatively small budget, so we'd like to avoid spending it on a large number of random initial points.


# %%
import os

import gpflow
from gpflow.utilities.traversal import read_values, multiple_assign
from gpflow.models import GPR
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from util.plotting import plot_regret

import trieste
from trieste.data import Dataset
from trieste.models.gpflow import GaussianProcessRegression
from trieste.models.optimizer import Optimizer
from trieste.objectives import TRID_10_MINIMUM, TRID_10_SEARCH_SPACE, trid_10
from trieste.objectives.utils import mk_observer
from trieste.space import Box
from trieste.models.normalization import DataTransformModelWrapper, MinMaxTransformer, StandardTransformer

np.random.seed(1794)
tf.random.set_seed(1794)


# %% [markdown]
#
# First, define the problem.

# %%
function = trid_10
F_MINIMUM = TRID_10_MINIMUM
search_space = TRID_10_SEARCH_SPACE

# %% [markdown]
# ## Collect initial points
#
# We set up the observer as usual over the Trid function search space, using Sobol sampling to sample the initial points.

# %%
num_initial_points = 15

observer = mk_observer(function)

initial_query_points = search_space.sample_sobol(num_initial_points)
initial_data = observer(initial_query_points)


# %% [markdown]
#
# Next, let's define the GP model.

# %%

def build_gp_model(data, x_std = 1.0, y_std = 0.1):

    dim = data.query_points.shape[-1]
    empirical_variance = tf.math.reduce_variance(data.observations)

    prior_lengthscales = [0.2*x_std*np.sqrt(dim)] * dim
    prior_scale = tf.cast(1.0, dtype=tf.float64)

    x_std = tf.cast(x_std, dtype=tf.float64)
    y_std = tf.cast(y_std, dtype=tf.float64)

    kernel = gpflow.kernels.Matern52(
        variance=empirical_variance,
        lengthscales=prior_lengthscales,
    )
    kernel.variance.prior = tfp.distributions.LogNormal(
        tf.math.log(y_std), prior_scale
    )
    kernel.lengthscales.prior = tfp.distributions.LogNormal(
        tf.math.log(kernel.lengthscales), prior_scale
    )
    gpr = gpflow.models.GPR(
        data.astuple(),
        kernel,
        mean_function=gpflow.mean_functions.Constant(),
        noise_variance=1e-5,
    )
    gpflow.set_trainable(gpr.likelihood, False)
    # gpflow.set_trainable(gpr.mean_function, False)
    # gpflow.set_trainable(gpr.kernel.kernels[1], False)

    return gpr

# %% [markdown]
#
# Build a wrapped model, that handles data transformation using only the initial dataset.

# %%

class GPRwithDataNormalization(DataTransformModelWrapper, GaussianProcessRegression):
    pass

# query_point_transformer simply transforms the search space to the unit cube.
query_point_transformer = MinMaxTransformer(tf.stack((search_space.lower, search_space.upper)))
observation_transformer = StandardTransformer(initial_data.observations)
model = GPRwithDataNormalization(
    dataset=initial_data,
    query_point_transformer=query_point_transformer,
    observation_transformer=observation_transformer,
    model=build_gp_model(initial_data),
    optimizer=Optimizer(
        gpflow.optimizers.Scipy(),
        minimize_args={"options": dict(maxiter=100)}
    ),
    num_kernel_samples=100,
)


# %% [markdown]
#
# Now we optimize and display the results.

# %%
num_acquisitions = 85

np.random.seed(1794)
tf.random.set_seed(1794)

bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
result = bo.optimize(num_acquisitions, initial_data, model)
dataset = result.try_get_final_dataset()

query_points = dataset.query_points.numpy()
observations = dataset.observations.numpy()

arg_min_idx = tf.squeeze(tf.argmin(observations, axis=0))

print(f"query point: {query_points[arg_min_idx, :]}")
print(f"observation: {observations[arg_min_idx, :]}")

def plot_regret_with_min(dataset):
    observations = dataset.observations.numpy()
    arg_min_idx = tf.squeeze(tf.argmin(observations, axis=0))

    suboptimality = observations - F_MINIMUM.numpy()
    ax = plt.gca()
    plot_regret(suboptimality, ax, num_init=num_initial_points, idx_best=arg_min_idx)

    ax.set_yscale("log")
    ax.set_ylabel("Regret")
    ax.set_ylim(0.001, 100000)
    ax.set_xlabel("# evaluations")

plot_regret_with_min(dataset)


# %% [markdown]
#
# Next, we add methods to update model and normalization parameters. Typically, the user will want to overload `_initialise_model_parameters` and `_update_model_parameters`.

# %%

class GPRwithDynamicDataNormalization(DataTransformModelWrapper, GaussianProcessRegression):
    """DataTransformWrapper for a GaussianProcessRegression model."""

    def _process_hyperparameter_dictionary(self, hyperparameters, inverse_transform = False):
        """Transform model hyperparameters based on data transforms.

        :param hyperparameters: The untransformed hyperparameters.
        :param inverse_transform: Whether to apply the forward transform (if False) or the inverse
            transform (if True).
        :returns: The transformed hyperparameters.
        """
        prefix = 'inverse_' if inverse_transform else ''
        processed_hyperparameters = {}
        for key, value in hyperparameters.items():
            tf_value = tf.constant(value, dtype=tf.float64)  # Ensure value is tf Tensor
            if key.endswith('mean_function.c'):
                transform = getattr(self._observation_transformer, f'{prefix}transform')
            elif key.endswith('variance') and 'likelihood' not in key and '[1]' not in key:
                transform = getattr(self._observation_transformer, f'{prefix}transform_variance')
            elif key.endswith('lengthscales') or key.endswith('period'):
                transform = getattr(self._query_point_transformer, f'{prefix}transform')
            else:
                transform = lambda x: x
            processed_hyperparameters[key] = transform(tf_value)
        return processed_hyperparameters

    def _transform_and_assign_hyperparameters(self, hyperparameters):
        """Transform hyperparameters for normalized data, and assign to model.

        :param hyperparameters: Hyperparameters for unnormalized data.
        """
        normalized_hyperparameters = self._process_hyperparameter_dictionary(hyperparameters)
        multiple_assign(self._model, normalized_hyperparameters)

    def _update_normalization_parameters(self, dataset: Dataset) -> None:
        """Update normalization parameters for the new dataset.

        :param dataset: New, unnormalized, dataset.
        """
        # self._query_point_transformer.set_parameters(dataset.query_points)
        # self._observation_transformer.set_parameters(dataset.observations)
        pass

    def _update_model_and_normalization_parameters(self, dataset):
        """Update the model and normalization parameters based on the new dataset.
        i.e. Denormalize using the old parameters, and renormalize using parameters set from
        the new dataset.

        :param dataset: New, unnormalized, dataset.
        """
        hyperparameters = read_values(self._model)
        print(hyperparameters['.mean_function.c'])
        unnormalized_hyperparameters = self._process_hyperparameter_dictionary(
            hyperparameters, inverse_transform=True
        )
        self._update_normalization_parameters(dataset)
        self._transform_and_assign_hyperparameters(unnormalized_hyperparameters)
        print(read_values(self._model)['.mean_function.c'])

query_point_transformer = MinMaxTransformer(tf.stack((search_space.lower, search_space.upper)))
observation_transformer = StandardTransformer(initial_data.observations)
dynamic_model = GPRwithDynamicDataNormalization(
    dataset=initial_data,
    query_point_transformer=query_point_transformer,
    observation_transformer=observation_transformer,
    update_parameters=True,
    model=build_gp_model(initial_data),
    optimizer=Optimizer(
        gpflow.optimizers.Scipy(),
        minimize_args={"options": dict(maxiter=100)}
    ),
    num_kernel_samples=100,
)


# %% [markdown]
#
# Let's optimize and display the results. Notice that these are better than above.

# %%
np.random.seed(1794)
tf.random.set_seed(1794)

bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
result = bo.optimize(num_acquisitions, initial_data, dynamic_model)
dataset = result.try_get_final_dataset()

query_points = dataset.query_points.numpy()
observations = dataset.observations.numpy()

arg_min_idx = tf.squeeze(tf.argmin(observations, axis=0))

print(f"query point: {query_points[arg_min_idx, :]}")
print(f"observation: {observations[arg_min_idx, :]}")

plot_regret_with_min(dataset)


# %% [markdown]
# ## LICENSE
#
# [Apache License 2.0](https://github.com/secondmind-labs/trieste/blob/develop/LICENSE)
