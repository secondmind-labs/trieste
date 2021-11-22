# %% [markdown]
# # Active Learning for Gaussian Process Classification Model

# %%
from dataclasses import astuple

import gpflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from gpflow.utilities import print_summary, set_trainable

import trieste
from trieste.acquisition.function import BayesianActiveLearningByDisagreement, PredictiveVariance
from trieste.acquisition.rule import OBJECTIVE
from trieste.models import create_model
from trieste.models.gpflow.models import GaussianProcessRegression, VariationalGaussianProcess
from trieste.objectives.utils import mk_observer
from trieste.utils import map_values

np.random.seed(1793)
tf.random.set_seed(1793)

# %% [markdown]
# ## The problem

# %% [markdown]
# In Trieste, it is also possible to query most interesting points for learning the problem, i.e we want to have as little data as possible to construct the best possible model (active learning). In this tutorial we will try to do active learning for binary classification problem using Bayesain Active Learning by Disagreement (BALD) for a Gaussian Process Classification Model.
#
# Our problem is classification problem of circle dataset. In this toturial we assume that the input space is continous so we can use continuous optimiser for our BALD acquisition function. 

# %%
search_space = trieste.space.Box([-1, -1], [1, 1])
input_dim = 2

def circle(x):
    return tf.cast((tf.reduce_sum(tf.square(x), axis=1, keepdims=True) - 0.5) > 0, tf.float64)


# %%
density = 100
xx = np.linspace(search_space.lower[0], search_space.upper[0], density)

grid_xx = np.meshgrid(*[xx] * input_dim)
xx = np.vstack([g.ravel() for g in grid_xx]).T
yy = circle(xx).numpy()

plt.figure(figsize=(5,5))
plt.contour(*grid_xx, np.reshape(yy, [density] * input_dim), levels=[0.5])
idx = np.squeeze(yy).astype(bool)
plt.scatter(xx[idx][:, 0], xx[idx][:, 1], label="1")
plt.scatter(xx[np.logical_not(idx)][:, 0], xx[np.logical_not(idx)][:, 1], label="0")
plt.legend()
plt.show()

# %% [markdown]
# Let's generate some data for our initial model. Here we randomly sample 10 data points.

# %%
numSamples = 10
X = search_space.sample(numSamples)
observer = mk_observer(circle, OBJECTIVE)
datasets = observer(X)

# %% [markdown]
# ## Modelling the binary classification task

# %% [markdown]
# For the binary classification model, we use the Variational Gaussian Process with Bernoulli likelihood. For more detail of this model, see BLABLA. 

# %%
from trieste.models.gpflow import GPflowModelConfig, VariationalGaussianProcess


def create_bo_model(data):
    kernel = gpflow.kernels.SquaredExponential(lengthscales=[0.2, 0.2])
    m = gpflow.models.VGP(astuple(data), likelihood=gpflow.likelihoods.Bernoulli(), kernel=kernel)
    return trieste.models.create_model(
        GPflowModelConfig(
            **{
                "model": m,
                "optimizer": gpflow.optimizers.Scipy(),
                "optimizer_args": {
                    "minimize_args": {"options": dict(maxiter=100.0)},
                },
            }
        )
    )

"""
# model seems harder to train using default VariationalGaussianProcess() optimiser which is adam?

def create_bo_model(data):
    kernel = gpflow.kernels.SquaredExponential()
    m = gpflow.models.VGP(astuple(data), likelihood=gpflow.likelihoods.Bernoulli(), kernel=kernel)
    return VariationalGaussianProcess(m)
"""

# %% [markdown]
# Lets see our model landscape using only those initial data

# %%
model = create_bo_model(datasets[OBJECTIVE])

model.update(datasets[OBJECTIVE])
model.optimize(datasets[OBJECTIVE])

mean, variance = model.predict(xx)

plt.figure()
plt.contourf(*grid_xx, np.reshape(mean, [density] * input_dim))
plt.plot(
    datasets[OBJECTIVE].query_points[:, 0],
    datasets[OBJECTIVE].query_points[:, 1],
    "ko",
    markersize=10,
)
plt.title("Mean")
plt.colorbar()
plt.show()

plt.figure()
plt.contourf(*grid_xx, np.reshape(variance, [density] * input_dim))
plt.colorbar()
plt.plot(
    datasets[OBJECTIVE].query_points[:, 0],
    datasets[OBJECTIVE].query_points[:, 1],
    "ko",
    markersize=10,
)
plt.title("Variance")
plt.show()


# %% [markdown]
# ## The acquisition process
#
# We can construct the BALD acqusition function which maximise information gain about the model parameters, by maximising the mutual information between predictions and model posterior:
#
# $$\mathbb{I}\left[y, \boldsymbol{\theta} \mid \mathbf{x}, \mathcal{D}\right]=\mathbb{H}\left[y \mid \mathbf{x}, \mathcal{D}\right]-\mathbb{E}_{p\left(\boldsymbol{\theta} \mid \mathcal{D}\right)}[\mathbb{H}[y \mid \mathbf{x}, \boldsymbol{\theta}]]$$
#
# See BLABLA for more details. Then, Trieste's `EfficientGlobalOptimization` is used for the query rule: 

# %%
initial_models = trieste.utils.map_values(create_bo_model, datasets)
acq = BayesianActiveLearningByDisagreement()
rule = trieste.acquisition.rule.EfficientGlobalOptimization(acq.using(OBJECTIVE))

# %% [markdown]
# ## Run the active learning loop
# Let's run our active learning iteration:

# %%
n_steps = 25
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
results = bo.optimize(n_steps, datasets, initial_models, rule, track_state=False)
final_dataset = results.try_get_final_datasets()[OBJECTIVE]
final_model = results.try_get_final_models()[OBJECTIVE]

# %% [markdown]
# ## Visualising the result
# Now, we can visualize our model after the active learning run

# %%
xmax = final_dataset.query_points[-n_steps:, :]

# %% Plot BO results
mean, variance = final_model.predict(xx)

def invlink(f):
    return gpflow.likelihoods.Bernoulli().invlink(f).numpy()

mean = invlink(mean)

plt.figure(figsize=(7,5))
plt.contourf(*grid_xx, np.reshape(mean, [density] * input_dim))
plt.colorbar()
plt.plot(
    final_dataset.query_points[:-n_steps, 0],
    final_dataset.query_points[:-n_steps, 1],
    "ko",
    markersize=10,
)
plt.plot(
    final_dataset.query_points[-n_steps:, 0], final_dataset.query_points[-n_steps:, 1], "rx", mew=10
)
plt.contour(*grid_xx, np.reshape(yy, [density] * input_dim), levels=[0.5])
plt.title("Updated Mean")
plt.show()

# %% [markdown]
# As expected, BALD will query in important regions like points near the domain boundary and class boundary. 
