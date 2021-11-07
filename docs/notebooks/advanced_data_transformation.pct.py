# %% [markdown]
# # Advanced Data Transformation

# %%
import os

import gpflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from util.plotting import plot_regret

import trieste
from trieste.data import Dataset
from trieste.models.gpflow import GaussianProcessRegression
from trieste.models.optimizer import Optimizer
from trieste.objectives.single_objectives import BEALE_MINIMUM, BEALE_SEARCH_SPACE, beale
from trieste.objectives.utils import mk_observer
from trieste.models.normalization import DataTransformModelWrapper, MinMaxTransformer, StandardTransformer

np.random.seed(1794)
tf.random.set_seed(1794)

# silence TF warnings and info messages, only print errors
# https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")


# %% [markdown]
# ## Describe the problem
#
# In advanced cases the user may want to update normalization and model parameters based on incoming data. This can be achieved by overloading the appropriate methods of :class:``DataTransformModelWrapper``.
#
# As an example, we will try to minimize the Beale function, which is multimodal and has extremely sharp peaks in the corners of its domain. We will start with a relatively small number of initial points, which is desirable when evaluations of the objective function are expensive. We will normalize the observations to zero mean and unit variance, and the query points to the unit cube. In this case, BO performance can be improved by updating the normalization parameters in light of new data, and updating the mean function commensurately. (Note that this is provided merely as an illustrative example; there is no consensus in the BO community about data normalization. The most common approach is to normalize with parameters based on the initial dataset only, and keep these fixed throughout.)

# %%
function = beale
F_MINIMUM = BEALE_MINIMUM
search_space = BEALE_SEARCH_SPACE

# %% [markdown]
# ## Collect initial points
#
# We set up the observer as usual and use a Sobol sequence to sample the initial points.

# %%
num_dims = search_space.lower.shape[0]
num_initial_points = num_dims + 2

observer = mk_observer(function)

initial_query_points = search_space.sample_sobol(num_initial_points)
initial_data = observer(initial_query_points)


# %% [markdown]
# ## Define the model
#
# Next, let's define the GP model.

# %%

def build_gp_model(data):

    dim = data.query_points.shape[-1]

    prior_lengthscales = [1.0] * dim
    prior_scale = tf.cast(1.0, dtype=tf.float64)

    kernel = gpflow.kernels.Matern52(
        variance=1.0,
        lengthscales=prior_lengthscales,
    )
    kernel.variance.prior = tfp.distributions.LogNormal(
        tf.math.log(kernel.variance), prior_scale
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

    return gpr

# %% [markdown]
# ## Normalization with static normalization parameters
#
# Build a wrapped model, that handles data transformation using only the initial dataset.

# %%
class GPRwithDataNormalization(DataTransformModelWrapper, GaussianProcessRegression):
    pass

# query_point_transformer simply transforms the search space to the unit cube.
query_point_transformer = MinMaxTransformer(tf.stack((search_space.lower, search_space.upper)))
observation_transformer = StandardTransformer(initial_data.observations)
normalized_data = Dataset(
    query_point_transformer.transform(initial_data.query_points),
    observation_transformer.transform(initial_data.observations)
)
model = GPRwithDataNormalization(
    model=build_gp_model(normalized_data),
    optimizer=Optimizer(
        gpflow.optimizers.Scipy(),
        minimize_args={"options": dict(maxiter=100)}
    ),
    num_kernel_samples=100,
    query_point_transformer=query_point_transformer,
    observation_transformer=observation_transformer,
)


# %% [markdown]
#
# Now we optimize and display the results.

# %%
num_acquisitions = 30 * num_dims - num_initial_points

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
# ## Updating normalization and model parameters every iteration
#
# We now show how to modify the above to add methods to update model and normalization parameters. To achieve this, the user simply needs to define a method ``_update_model_and_normalization_parameters``.

# %%
class GPRwithDynamicDataNormalization(DataTransformModelWrapper, GaussianProcessRegression):
    """DataTransformWrapper for a GaussianProcessRegression model."""

    def _update_model_and_normalization_parameters(self, dataset):
        """Update the model and normalization parameters based on the new dataset.
        i.e. Denormalize using the old parameters, and renormalize using parameters set from
        the new dataset.

        :param dataset: New, unnormalized, dataset.
        """
        unnormalized_mean = self._observation_transformer.inverse_transform(self._model.mean_function.c)
        self._observation_transformer.set_parameters(dataset.observations)
        self._model.mean_function.c.assign(self._observation_transformer.transform(unnormalized_mean))


# %% [markdown]
#
# Now we proceed as usual, building the wrapped model, running BO, and displaying the results.

# %%
query_point_transformer = MinMaxTransformer(tf.stack((search_space.lower, search_space.upper)))
observation_transformer = StandardTransformer(initial_data.observations)
normalized_data = Dataset(
    query_point_transformer.transform(initial_data.query_points),
    observation_transformer.transform(initial_data.observations)
)
dynamic_model = GPRwithDynamicDataNormalization(
    model=build_gp_model(normalized_data),
    optimizer=Optimizer(
        gpflow.optimizers.Scipy(),
        minimize_args={"options": dict(maxiter=100)}
    ),
    num_kernel_samples=100,
    query_point_transformer=query_point_transformer,
    observation_transformer=observation_transformer,
)

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
