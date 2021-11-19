# %% [markdown]
# # Bayesian active learning of feasible sets

# %% [markdown]
#

# %%
# %matplotlib inline
import numpy as np
import tensorflow as tf

np.random.seed(1793)
tf.random.set_seed(1793)

# %% [markdown]
# ## Describe the problem
#
# In this example, we will perform active learning for the scaled Branin function.


# %%
from trieste.objectives import scaled_branin
from util.plotting_plotly import plot_function_plotly
from trieste.space import Box

search_space = Box([0, 0], [1, 1])

# fig = plot_function_plotly(scaled_branin, search_space.lower, search_space.upper, grid_density=20)
# fig.update_layout(height=400, width=400)
# fig.show()

# %% [markdown]
# We begin our Bayesian active learning from a small initial design built from a space-filling Halton sequence.

# %%
import trieste

observer = trieste.objectives.utils.mk_observer(scaled_branin)

num_initial_points = 12
initial_query_points = search_space.sample_halton(num_initial_points)
initial_data = observer(initial_query_points)


# %% [markdown]
# ## Surrogate model
#
# Just like in sequential optimization, we fit a surrogate Gaussian process model to the initial data.

# %%
import gpflow
from trieste.models.gpflow.models import GaussianProcessRegression


def build_model(data):
    variance = tf.math.reduce_variance(data.observations)
    kernel = gpflow.kernels.Matern32(variance=variance, lengthscales=[.2, .2])
    gpr = gpflow.models.GPR(data.astuple(), kernel, noise_variance=1e-5)
    gpflow.set_trainable(gpr.likelihood, False)
    return GaussianProcessRegression(gpr)


model = build_model(initial_data)

# %% [markdown]
# ## Active learning
#

# %%
from trieste.acquisition.optimizer import generate_continuous_optimizer
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.acquisition.function import IntegratedVarianceReduction

threshold = [.3, .7]
num_integral_points = 1000

integration_points = search_space.sample_sobol(num_integral_points)
acq = IntegratedVarianceReduction(threshold=threshold, integration_points=integration_points)
rule = EfficientGlobalOptimization(builder=acq)
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

from util.plotting import plot_acq_function_2d
import matplotlib.pyplot as plt
acq_function = acq.prepare_acquisition_function(model=model, dataset=initial_data)
plot_acq_function_2d(acq_function, search_space.lower, search_space.upper, contour=True, grid_density=100, colorbar=True)
plt.scatter(
    initial_data.query_points[:, 0].numpy(),
    initial_data.query_points[:, 1].numpy(),
    color="red",
    lw=5,
    marker="*",
    zorder=1,
)
plt.show()
# %% [markdown]
# To plot the contour of variance of our model at each step, we can set the `track_state` parameter to `True` in `bo.optimize()`, this will make trieste record our model at each iteration.

# %%
bo_iter = 30
result = bo.optimize(bo_iter, initial_data, model, rule, track_state=True)

from trieste.acquisition import ExpectedImprovement

# plot standard EI acquisition function



# %% [markdown]
# Then we can retrieve our final dataset from the active learning steps.

# %%
dataset = result.try_get_final_dataset()
query_points = dataset.query_points.numpy()
observations = dataset.observations.numpy()

# %% [markdown]
# Finally, we can check the performance of

# %%
from util.plotting import plot_bo_points, plot_function_2d
import tensorflow_probability as tfp

def excursion_probability_lower(x):
    mean, variance = result.history[-1].models["OBJECTIVE"].model.predict_f(x)
    normal = tfp.distributions.Normal(mean, tf.sqrt(variance))
    return normal.cdf(threshold[0])

def excursion_probability_upper(x):
    mean, variance = result.history[-1].models["OBJECTIVE"].model.predict_f(x)
    normal = tfp.distributions.Normal(mean, tf.sqrt(variance))
    return normal.cdf(threshold[1])

def excursion_probability_band(x):
    mean, variance = result.history[-1].models["OBJECTIVE"].model.predict_f(x)
    normal = tfp.distributions.Normal(mean, tf.sqrt(variance))
    return normal.cdf(threshold[1]) - normal.cdf(threshold[0])

fig, ax = plot_function_2d(
    excursion_probability_lower,
    search_space.lower - 0.01,
    search_space.upper + 0.01,
    grid_density=100,
    contour=True,
    colorbar=True,
    figsize=(10, 6),
    title="Variance contour with queried points at iter:",
    xlabel="$X_1$",
    ylabel="$X_2$",
)

plot_bo_points(
    query_points, ax[0, 0], num_initial_points
)
fig.show()

fig, ax = plot_function_2d(
    excursion_probability_upper,
    search_space.lower - 0.01,
    search_space.upper + 0.01,
    grid_density=100,
    contour=True,
    colorbar=True,
    figsize=(10, 6),
    title="Variance contour with queried points at iter:",
    xlabel="$X_1$",
    ylabel="$X_2$",
)

plot_bo_points(
    query_points, ax[0, 0], num_initial_points
)
fig.show()

fig, ax = plot_function_2d(
    excursion_probability_band,
    search_space.lower - 0.01,
    search_space.upper + 0.01,
    grid_density=100,
    contour=True,
    colorbar=True,
    figsize=(10, 6),
    title="Variance contour with queried points at iter:",
    xlabel="$X_1$",
    ylabel="$X_2$",
    alpha=0.5
)

plot_bo_points(
    query_points, ax[0, 0], num_initial_points, c_pass="tab:red",
)
fig.show()
# %% [markdown]
# ## LICENSE
#
# [Apache License 2.0](https://github.com/secondmind-labs/trieste/blob/develop/LICENSE)
