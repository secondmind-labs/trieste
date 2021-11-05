# %% [markdown]
# # Advanced Data Transformation

# %%
#
# In advanced cases the user may want to update normalization and model parameters based on incoming data. This can be achieved by subclassing `DataTransformWrapper`.
#
# First, we subclass DataTransformWrapper, and define methods to allow updating the model parameters.
#


# %%
import os

import gpflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from util.plotting import plot_regret

import trieste
from trieste.ask_tell_optimization import AskTellOptimizer
from trieste.data import Dataset
from trieste.models.gpflow import GaussianProcessRegression
from trieste.models.optimizer import Optimizer
from trieste.objectives import TRID_10_MINIMUM, TRID_10_SEARCH_SPACE, trid_10
from trieste.objectives.utils import mk_observer
from trieste.space import Box


# %%
function = trid_10
F_MINIMUM = TRID_10_MINIMUM
search_space = TRID_10_SEARCH_SPACE


# ## Collect initial points
#
# We set up the observer as usual over the Trid function search space, using Sobol sampling to sample the initial points.

# %%
num_initial_points = 50

observer = mk_observer(function)

initial_query_points = search_space.sample_sobol(num_initial_points)
initial_data = observer(initial_query_points)

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

    return gpr

gp_model = build_gp_model(initial_data)

# %%
class GPRwithNormalizedData(DataTransformWrapper, GaussianProcessRegression):
    """DataTransformWrapper for a GaussianProcessRegression model."""
    def __init__(self, model: GPR | SGPR, optimizer: Optimizer | None = None, num_kernel_samples: int = 10):
        super().__init__(model, optimizer=optimizer, num_kernel_samples=num_kernel_samples)
        # Assume that model hyperparameters were defined for normalized data.

    def _process_hyperparameter_dictionary(self, hyperparameters, inverse_transform):
        """Transform model hyperparameters based on data transforms.

        :param hyperparameters: The untransformed hyperparameters.
        :param inverse_transform: Whether to apply the forward transform (if False) or the inverse
            transform (if True).
        :returns: The transformed hyperparameters.
        """
        prefix = 'inverse_' if inverse_transform else ''
        processed_hyperparameters = {}
        for key, value in hyperparameters.items():
            tf_value = tf.constant(value, dtype=default_float())  # Ensure value is tf Tensor
            if key.endswith('mean_function.c'):
                transform = getattr(self._observation_transformer, f'{prefix}transform')
            elif key.endswith('variance') and not 'likelihood' in key:
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

    def _initialize_model_parameters(self):
        """Update initial model hyperparameters by transforming into normalized space."""
        hyperparameters = read_values(self._model)
        hyperparameters = {k: tf.constant(v, dtype=default_float()) for k, v in hyperparameters.items()}
        self._transform_and_assign_hyperparameters(hyperparameters)

    def _update_model_and_normalization_parameters(self, dataset):
        """Update the model and normalization parameters based on the new dataset.
        i.e. Denormalize using the old parameters, and renormalize using parameters set from
        the new dataset.

        :param dataset: New, unnormalized, dataset.
        """
        hyperparameters = read_values(self._model)
        unnormalized_hyperparameters = self._process_hyperparameter_dictionary(
            hyperparameters, inverse_transform=True
        )
        unnormalized_hyperparameter_priors = self._get_unnormalised_hyperparameter_priors()
        self._update_normalization_parameters(dataset)
        self._transform_and_assign_hyperparameters(unnormalized_hyperparameters)
        self._update_hyperparameter_priors(unnormalized_hyperparameter_priors)


# %% [markdown]
#
# Next, we define our query point and observation transformers, and run optimization as usual.

# %%
# Create transformers and pass these in.
query_point_transformer = StandardTransformer(initial_data.query_points)
observation_transformer = StandardTransformer(initial_data.observations)
model = GPRwithDataNormalization(
    dataset=initial_data,
    query_point_transformer=query_point_transformer,
    observation_transformer=observation_transformer,
    model=gp_model,
    optimizer=Optimizer(
        gpflow.optimizers.Scipy(),
        minimize_args={"options": dict(maxiter=100)}
    ),
    num_kernel_samples=100,
)

# %%
num_acquisitions = 100
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
result = bo.optimize(num_acquisitions, initial_data, model)
dataset = result.try_get_final_dataset()

query_points = dataset.query_points.numpy()
observations = dataset.observations.numpy()

arg_min_idx = tf.squeeze(tf.argmin(observations, axis=0))

plot_regret_with_min(dataset)

print(f"query point: {query_points[arg_min_idx, :]}")
print(f"observation: {observations[arg_min_idx, :]}")


# %%
#
# Let's have a look at the results.
