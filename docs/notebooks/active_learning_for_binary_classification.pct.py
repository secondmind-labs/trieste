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
# We will illustrate the BALD algorithm on a synthetic binary classification problem where one class takes shape of a circle in the search space. The input space is continuous so we can use continuous optimiser for our BALD acquisition function.

# %%
search_space = trieste.space.Box([-1, -1], [1, 1])
input_dim = 2


def circle(x):
    return tf.cast((tf.reduce_sum(tf.square(x), axis=1, keepdims=True) - 0.5) > 0, tf.float64)


# %% [markdown]
# Let's first illustrate how this two dimensional problem looks like.

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
# Let's generate some data for our initial model. Here we randomly sample a small number of data points.

# %%
num_initial_points = 5
X = search_space.sample(num_initial_points)
observer = mk_observer(circle)
initial_data = observer(X)

# %% [markdown]
# ## Modelling the binary classification task

# %% [markdown]
# For the binary classification model, we use the Variational Gaussian Process with Bernoulli likelihood. For more detail of this model, see <cite data-cite="Nickisch08a">[Nickisch et al.](https://www.jmlr.org/papers/volume9/nickisch08a/nickisch08a.pdf)</cite>. Here we use trieste's gpflow model builder `build_vgp_classifier`.
# User can also use Sparse Variational Gaussian Process(SVGP) for building the classification model via `build_svgp` function and `SparseVariational` class. SVGP is preferable for bigger amount of data.

# %%
from trieste.models.gpflow import VariationalGaussianProcess
from trieste.models.gpflow.builders import build_vgp_classifier

model = VariationalGaussianProcess(
    build_vgp_classifier(initial_data, search_space, noise_free=True)
)

# %% [markdown]
# Lets see our model landscape using only those initial data

# %%
from util.plotting import plot_bo_points, plot_function_2d

model.update(initial_data)
model.optimize(initial_data)


def plot_active_learning_query(function, points, num_initial_points, title):
    _, ax = plot_function_2d(
        function,
        search_space.lower,
        search_space.upper,
        grid_density=100,
        contour=True,
        colorbar=True,
        title=[title],
        xlabel="$X_1$",
        ylabel="$X_2$",
    )

    plot_bo_points(
        points,
        ax[0, 0],
        num_initial_points,
    )


def pred_var(x):
    _, var = model.predict(x)
    return var


def pred_mean(x):
    mean, _ = model.predict(x)
    return mean


plot_active_learning_query(pred_mean, X, num_initial_points, title="Mean")
plot_active_learning_query(pred_var, X, num_initial_points, title="Variance")


# %% [markdown]
# ## The acquisition process
#
# We can construct the BALD acquisition function which maximises information gain about the model parameters, by maximising the mutual information between predictions and model posterior:
#
# $$\mathbb{I}\left[y, \boldsymbol{\theta} \mid \mathbf{x}, \mathcal{D}\right]=\mathbb{H}\left[y \mid \mathbf{x}, \mathcal{D}\right]-\mathbb{E}_{p\left(\boldsymbol{\theta} \mid \mathcal{D}\right)}[\mathbb{H}[y \mid \mathbf{x}, \boldsymbol{\theta}]]$$
#
# See <cite data-cite="houlsby2011bayesian">[Houlsby et al.](https://arxiv.org/pdf/1112.5745.pdf)</cite> for more details. Then, Trieste's `EfficientGlobalOptimization` is used for the query rule:

# %%
acq = BayesianActiveLearningByDisagreement()
rule = trieste.acquisition.rule.EfficientGlobalOptimization(acq)  # type: ignore

# %% [markdown]
# ## Run the active learning loop
# Let's run our active learning iteration:

# %%
n_steps = 30
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
results = bo.optimize(n_steps, initial_data, model, rule, track_state=False)
final_dataset = results.try_get_final_datasets()[OBJECTIVE]
final_model = results.try_get_final_models()[OBJECTIVE]

# %% [markdown]
# ## Visualising the result
# Now, we can visualize our model after the active learning run

# %% Plot BO results
mean, variance = final_model.predict(query_points)

mean = gpflow.likelihoods.Bernoulli().invlink(mean).numpy()

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
