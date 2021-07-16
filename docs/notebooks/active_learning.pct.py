# %% [markdown]
# # Active Learning

# %% [markdown]
# Sometimes rather than optimizing a black-box function, we want to learn the function better by reducing our model uncertainty. The one way to do this is by doing sampling in unknown inputs regions (active learning). This notebook demonstrates to perform Bayesian active learning using `trieste`.

# %%
# %matplotlib inline
import numpy as np
import tensorflow as tf
import pandas as pd

np.random.seed(1793)
tf.random.set_seed(1793)

# %% [markdown]
# ## Describe the problem
#
# In this example, we will consider doing active learning for the log branin function.


# %%
from trieste.utils.objectives import branin
from util.plotting_plotly import plot_function_plotly
from trieste.space import Box


def log_branin(x):
    return tf.math.log(branin(x))


search_space = Box([0, 0], [1, 1])

fig = plot_function_plotly(log_branin, search_space.lower, search_space.upper, grid_density=20)
fig.update_layout(height=400, width=400)
fig.show()

# %% [markdown]
# We begin our Bayesian active learning steps after evaluating two initial points sampled by Halton sequence.

# %%
import trieste

observer = trieste.utils.objectives.mk_observer(log_branin)

num_initial_points = 2
initial_query_points = search_space.sample_halton(num_initial_points)
initial_data = observer(initial_query_points)


# %% [markdown]
# ## Surrogate model
#
# Just like in sequential optimization, we fit a surrogate Gaussian process model to the initial data.

# %%
import gpflow


def build_model(data):
    variance = tf.math.reduce_variance(data.observations)
    kernel = gpflow.kernels.RBF(variance=variance, lengthscales=[0.2, 0.2])
    gpr = gpflow.models.GPR(data.astuple(), kernel, noise_variance=1e-5)
    gpflow.set_trainable(gpr.likelihood, False)

    return {
        "model": gpr,
        "optimizer": gpflow.optimizers.Scipy(),
        "optimizer_args": {
            "minimize_args": {"options": dict(maxiter=100)},
        },
    }


model = build_model(initial_data)

# %% [markdown]
# ## Active learning using predictive variance
#
# To do our first active learning example, we will use `PredictiveVariance`, one of the simplest active learning acquisition function. the `PredictiveVariance` sample a point on highest determinant of the predictive covariance of our model. Here, we will consider single query point using `PredictiveVariance`, the batch case will be shown later. We can utilize trieste's `BayesianOptimizer` to do the active learning steps.

# %%
from trieste.acquisition.optimizer import generate_continuous_optimizer
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.acquisition.function import PredictiveVariance

acq = PredictiveVariance()
rule = EfficientGlobalOptimization(
    builder=acq, optimizer=generate_continuous_optimizer(sigmoid=False)
)
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

# %% [markdown]
# To plot the contour of variance of our model at each step, we can set the `track_state` parameter to `True` in `bo.optimize()`, this will make trieste record our model at each iteration.

# %%
bo_iter = 5
result = bo.optimize(bo_iter, initial_data, model, rule, track_state=True)

# %% [markdown]
# Then we can retrieve our final dataset from the active learning steps

# %%
dataset = result.try_get_final_dataset()
query_points = dataset.query_points.numpy()
observations = dataset.observations.numpy()

# %% [markdown]
# Finally, we can check the performance of our `PredictiveVariance` active learning acquisition function by plotting the predictive variance landscape of our model. We can see how it samples on regions with high variance value.

# %%
from util.plotting import plot_bo_points, plot_function_2d

for i in range(bo_iter):

    def pred_var(x):
        _, var = result.history[i].models["OBJECTIVE"].model.predict_f(x)
        return var

    _, ax = plot_function_2d(
        pred_var,
        search_space.lower - 0.01,
        search_space.upper + 0.01,
        grid_density=20,
        contour=True,
        colorbar=True,
        figsize=(10, 6),
        title=["Variance contour with queried points at iter:" + str(i + 1)],
        xlabel="$X_1$",
        ylabel="$X_2$",
    )
    plot_bo_points(query_points[: num_initial_points + i], ax[0, 0], num_initial_points)


# %% [markdown]
# ## Batch active learning using predictive variance
#
# For batch active learning using `PredictiveVariance`, we can set the `num_query_points` with the size of batch we want on the `EfficientGLobalOptimization` rule. The rest step is the same with the single query point of `PredictiveVariance` acquisition function.

# %%
bo_iter = 5
num_query = 3
acq = PredictiveVariance()
rule = EfficientGlobalOptimization(
    num_query_points=num_query, builder=acq, optimizer=generate_continuous_optimizer(sigmoid=False)
)
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

result = bo.optimize(bo_iter, initial_data, model, rule, track_state=True)
dataset = result.try_get_final_dataset()

query_points = dataset.query_points.numpy()
observations = dataset.observations.numpy()

from util.plotting import plot_bo_points, plot_function_2d

for i in range(bo_iter):

    def pred_var(x):
        _, var = result.history[i].models["OBJECTIVE"].model.predict_f(x)
        return var

    _, ax = plot_function_2d(
        pred_var,
        search_space.lower - 0.01,
        search_space.upper + 0.01,
        grid_density=20,
        contour=True,
        colorbar=True,
        figsize=(10, 6),
        title=["Variance contour with queried points at iter:" + str(i + 1)],
        xlabel="$X_1$",
        ylabel="$X_2$",
    )
    plot_bo_points(
        query_points[: num_initial_points + (i * num_query)], ax[0, 0], num_initial_points
    )
