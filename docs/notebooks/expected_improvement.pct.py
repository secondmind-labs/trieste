# %% [markdown]
# # Noise-free optimization with Expected Improvement

# %%
import numpy as np
import tensorflow as tf

np.random.seed(1793)
tf.random.set_seed(1793)

# %% [markdown]
# ## Describe the problem
# In this example, we look to find the minimum value of the two-dimensional Branin function over the hypercube $[0, 1]^2$. We can represent the search space using a `Box`, and plot contours of the Branin over this space.
#
#

# %%
from trieste.utils.objectives import branin
from util.plotting_plotly import plot_function_plotly
from trieste.space import Box

search_space = Box([0, 0], [1, 1])

fig = plot_function_plotly(branin, search_space.lower, search_space.upper, grid_density=20)
fig.update_layout(height=400, width=400)
fig.show()

# %% [markdown]
# ## Sample the observer over the search space
#
# Sometimes we don't have direct access to the objective function. We only have an observer that indirectly observes it. In _Trieste_, an observer can output a number of datasets. In our case, we only have one dataset, the objective. We can convert a function with `branin`'s signature to a single-output observer using `mk_observer`.
#
# The optimization procedure will benefit from having some starting data from the objective function to base its search on. We sample five points from the search space and evaluate them on the observer.

# %%
import trieste

observer = trieste.utils.objectives.mk_observer(branin)

num_initial_points = 5
initial_query_points = search_space.sample(num_initial_points)
initial_data = observer(initial_query_points)

# %% [markdown]
# ## Model the objective function
#
# The Bayesian optimization procedure estimates the next best points to query by using a probabilistic model of the objective. We'll use Gaussian process regression for this, provided by GPflow. The model will need to be trained on each step as more points are evaluated, so we'll package it with GPflow's Scipy optimizer.

# %%
import gpflow


def build_model(data):
    variance = tf.math.reduce_variance(data.observations)
    kernel = gpflow.kernels.Matern52(variance=variance, lengthscales=[0.2, 0.2])
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
# ## Run the optimization loop
#
# We can now run the Bayesian optimization loop by defining a `BayesianOptimizer` and calling its `optimize` method.
#
# The optimizer uses an acquisition rule to choose where in the search space to try on each optimization step. We'll use the default acquisition rule, which is Efficient Global Optimization with Expected Improvement.
#
# We'll run the optimizer for fifteen steps.
#
# The optimization loop catches errors so as not to lose progress, which means the optimization loop might not complete and the data from the last step may not exist. Here we'll handle this crudely by asking for the data regardless, using `.try_get_final_datasets()`, which will re-raise the error if one did occur. For a review of how to handle errors systematically, there is a [dedicated tutorial](recovering_from_errors.ipynb).

# %%
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

result = bo.optimize(15, initial_data, model)
dataset = result.try_get_final_dataset()

# %% [markdown]
# ## Explore the results
#
# We can now get the best point found by the optimizer. Note this isn't necessarily the point that was last evaluated.

# %%
query_points = dataset.query_points.numpy()
observations = dataset.observations.numpy()

arg_min_idx = tf.squeeze(tf.argmin(observations, axis=0))

print(f"query point: {query_points[arg_min_idx, :]}")
print(f"observation: {observations[arg_min_idx, :]}")

# %% [markdown]
# We can visualise how the optimizer performed by plotting all the acquired observations, along with the true function values and optima, either in a two-dimensional contour plot ...

# %%
from util.plotting import plot_bo_points, plot_function_2d

_, ax = plot_function_2d(
    branin, search_space.lower, search_space.upper, grid_density=30, contour=True
)
plot_bo_points(query_points, ax[0, 0], num_initial_points, arg_min_idx)

# %% [markdown]
# ... or as a three-dimensional plot

# %%
from util.plotting_plotly import add_bo_points_plotly

fig = plot_function_plotly(branin, search_space.lower, search_space.upper, grid_density=20)
fig.update_layout(height=500, width=500)

fig = add_bo_points_plotly(
    x=query_points[:, 0],
    y=query_points[:, 1],
    z=observations[:, 0],
    num_init=num_initial_points,
    idx_best=arg_min_idx,
    fig=fig,
)
fig.show()

# %% [markdown]
# We can also visualise the how each successive point compares the current best.
#
# We produce two plots. The left hand plot shows the observations (crosses and dots), the current best (orange line), and the start of the optimization loop (blue line). The right hand plot is the same as the previous two-dimensional contour plot, but without the resulting observations. The best point is shown in each (purple dot).

# %%
import matplotlib.pyplot as plt
from util.plotting import plot_regret

_, ax = plt.subplots(1, 2)
plot_regret(observations, ax[0], num_init=num_initial_points, idx_best=arg_min_idx)
plot_bo_points(query_points, ax[1], num_init=num_initial_points, idx_best=arg_min_idx)

# %% [markdown]
# We can visualise the model over the objective function by plotting the mean and 95% confidence intervals of its predictive distribution. Like with the data before, we can get the model with `.try_get_final_model()`.

# %%
from util.plotting_plotly import plot_gp_plotly

fig = plot_gp_plotly(
    result.try_get_final_model().model,  # type: ignore
    search_space.lower,
    search_space.upper,
    grid_density=30,
)

fig = add_bo_points_plotly(
    x=query_points[:, 0],
    y=query_points[:, 1],
    z=observations[:, 0],
    num_init=num_initial_points,
    idx_best=arg_min_idx,
    fig=fig,
    figrow=1,
    figcol=1,
)

fig.show()

# %% [markdown]
# We can also inspect the model hyperparameters, and use the history to see how the length scales evolved over iterations. Note the history is saved at the *start* of each step, and as such never includes the final result, so we'll add that ourselves.

# %%
gpflow.utilities.print_summary(
    result.try_get_final_model().model  # type: ignore
)

ls_list = [
    step.model.model.kernel.lengthscales.numpy()  # type: ignore
    for step in result.history + [result.final_result.unwrap()]
]

ls = np.array(ls_list)
plt.plot(ls[:, 0])
plt.plot(ls[:, 1])

# %% [markdown]
# ## Run the optimizer for more steps
#
# If we need more iterations for better convergence, we can run the optimizer again using the data produced from the last run, as well as the model. We'll visualise the final data.

# %%
result = bo.optimize(5, result.try_get_final_dataset(), result.try_get_final_model())
dataset = result.try_get_final_dataset()

arg_min_idx = tf.squeeze(tf.argmin(dataset.observations, axis=0))
_, ax = plot_function_2d(
    branin, search_space.lower, search_space.upper, grid_density=40, contour=True
)

plot_bo_points(
    dataset.query_points.numpy(),
    ax=ax[0, 0],
    num_init=len(dataset.query_points),
    idx_best=arg_min_idx,
)

# %% [markdown]
# ## LICENSE
#
# [Apache License 2.0](https://github.com/secondmind-labs/trieste/blob/develop/LICENSE)
