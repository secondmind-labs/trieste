# %% [markdown]
# # Active Learning for Gaussian Process Classification Model

# %%
import gpflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import trieste
from trieste.acquisition.function import BayesianActiveLearningByDisagreement
from trieste.acquisition.rule import OBJECTIVE
from trieste.models.gpflow.models import VariationalGaussianProcess
from trieste.objectives.utils import mk_observer

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
query_points = np.linspace(search_space.lower[0], search_space.upper[0], density)

grid_query_points = np.meshgrid(*[query_points] * input_dim)
query_points = np.vstack([g.ravel() for g in grid_query_points]).T
observations = circle(query_points).numpy()

plt.figure(figsize=(5, 5))
plt.contour(*grid_query_points, np.reshape(observations, [density] * input_dim), levels=[0.5])
idx = np.squeeze(observations).astype(bool)
plt.scatter(query_points[idx][:, 0], query_points[idx][:, 1], label="1")
plt.scatter(
    query_points[np.logical_not(idx)][:, 0], query_points[np.logical_not(idx)][:, 1], label="0"
)
plt.legend()
plt.show()

# %% [markdown]
# Let's generate some data for our initial model. Here we randomly sample 10 data points.

# %%
num_initial_points = 10
X = search_space.sample(num_initial_points)
observer = mk_observer(circle)
initial_data = observer(X)

# %% [markdown]
# ## Modelling the binary classification task

# %% [markdown]
# For the binary classification model, we use the Variational Gaussian Process with Bernoulli likelihood. For more detail of this model, see <cite data-cite="Nickisch08a">[Nickisch et al.](https://www.jmlr.org/papers/volume9/nickisch08a/nickisch08a.pdf)</cite>.

# %%
from trieste.models.gpflow import VariationalGaussianProcess
from trieste.models.optimizer import Optimizer

optimizer = Optimizer(
    optimizer=gpflow.optimizers.Scipy(), minimize_args={"options": dict(maxiter=100)}
)


def create_bo_model(data):
    kernel = gpflow.kernels.SquaredExponential(lengthscales=[0.2, 0.2])
    m = gpflow.models.VGP(data.astuple(), likelihood=gpflow.likelihoods.Bernoulli(), kernel=kernel)
    return VariationalGaussianProcess(m, optimizer)


# %% [markdown]
# Lets see our model landscape using only those initial data

# %%
model = create_bo_model(initial_data)

model.update(initial_data)
model.optimize(initial_data)

mean, variance = model.predict(query_points)

plt.figure()
plt.contourf(*grid_query_points, np.reshape(mean, [density] * input_dim))
plt.plot(
    initial_data.query_points[:, 0],
    initial_data.query_points[:, 1],
    "ko",
    markersize=10,
)
plt.title("Mean")
plt.colorbar()
plt.show()

plt.figure()
plt.contourf(*grid_query_points, np.reshape(variance, [density] * input_dim))
plt.colorbar()
plt.plot(
    initial_data.query_points[:, 0],
    initial_data.query_points[:, 1],
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
# See <cite data-cite="houlsby2011bayesian">[Houlsby et al.](https://arxiv.org/pdf/1112.5745.pdf)</cite> for more details. Then, Trieste's `EfficientGlobalOptimization` is used for the query rule:

# %%
initial_models = create_bo_model(initial_data)
acq = BayesianActiveLearningByDisagreement()
rule = trieste.acquisition.rule.EfficientGlobalOptimization(acq)

# %% [markdown]
# ## Run the active learning loop
# Let's run our active learning iteration:

# %%
n_steps = 25
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
results = bo.optimize(n_steps, initial_data, initial_models, rule, track_state=False)
final_dataset = results.try_get_final_datasets()[OBJECTIVE]
final_model = results.try_get_final_models()[OBJECTIVE]

# %% [markdown]
# ## Visualising the result
# Now, we can visualize our model after the active learning run

# %% Plot BO results
mean, variance = final_model.predict(query_points)


def invlink(f):
    return gpflow.likelihoods.Bernoulli().invlink(f).numpy()


mean = invlink(mean)

plt.figure(figsize=(7, 5))
plt.contourf(*grid_query_points, np.reshape(mean, [density] * input_dim))
plt.colorbar()
plt.plot(
    final_dataset.query_points[:-n_steps, 0],
    final_dataset.query_points[:-n_steps, 1],
    "ko",
    markersize=10,
    label="Initial points",
)
plt.plot(
    final_dataset.query_points[-n_steps:, 0],
    final_dataset.query_points[-n_steps:, 1],
    "rx",
    mew=10,
    label="queried points",
)
plt.contour(*grid_query_points, np.reshape(observations, [density] * input_dim), levels=[0.5])
plt.title("Updated Mean")
plt.legend()
plt.show()

# %% [markdown]
# As expected, BALD will query in important regions like points near the domain boundary and class boundary.

# %% [markdown]
# ## LICENSE
#
# [Apache License 2.0](https://github.com/secondmind-labs/trieste/blob/develop/LICENSE)
