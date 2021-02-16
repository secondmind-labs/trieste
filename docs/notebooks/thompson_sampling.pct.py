# %% [markdown]
# # Batch-sequential optimization with Thompson sampling

# %%
import numpy as np
import tensorflow as tf

np.random.seed(1793)
tf.random.set_seed(1793)

# %% [markdown]
# ## Define the problem and model
#
# You can use Thompson sampling for Bayesian optimization in much the same way as we used EGO and EI in the tutorial _Introduction_. Since the setup is much the same is in that tutorial, we'll skip over most of the detail.
#
# We'll use a continuous bounded search space, and evaluate the observer at ten random points.

# %%
import trieste
from trieste.utils.objectives import branin
from trieste.acquisition.rule import OBJECTIVE

search_space = trieste.space.Box([0, 0], [1, 1])

num_initial_data_points = 10
initial_query_points = search_space.sample(num_initial_data_points)
observer = trieste.utils.objectives.mk_observer(branin, OBJECTIVE)
initial_data = observer(initial_query_points)

# %% [markdown]
# We'll use Gaussian process regression to model the function.

# %%
import gpflow

observations = initial_data[OBJECTIVE].observations
kernel = gpflow.kernels.Matern52(tf.math.reduce_variance(observations), [0.2, 0.2])
gpr = gpflow.models.GPR(
    initial_data[OBJECTIVE].astuple(), kernel, noise_variance=1e-5
)
gpflow.set_trainable(gpr.likelihood, False)

model_config = {
    OBJECTIVE: {
        "model": gpr,
        "optimizer": gpflow.optimizers.Scipy(),
        "optimizer_args": {
            "minimize_args": {"options": dict(maxiter=100)},
        },
    }
}

# %% [markdown]
# ## Create the Thompson sampling acquisition rule
#
# We achieve Bayesian optimization with Thompson sampling by specifying `ThompsonSampling` as the acquisition rule. Unlike the `EfficientGlobalOptimization` acquisition rule, `ThompsonSampling` does not use an acquisition function. Instead, in each optimization step, the rule samples `num_query_points` samples from the model posterior at `num_search_space_samples` points on the search space. It then returns the `num_query_points` points of those that minimise the model posterior.

# %%
acq_rule = trieste.acquisition.rule.ThompsonSampling(
    num_search_space_samples=1000, num_query_points=10
)

# %% [markdown]
# ## Run the optimization loop
#
# All that remains is to pass the Thompson sampling rule to the `BayesianOptimizer`. Once the optimization loop is complete, the optimizer will return `num_query_points` new query points for every step in the loop. With five steps, that's fifty points.

# %%
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

result = bo.optimize(5, initial_data, model_config, acq_rule, track_state=False)
dataset = result.try_get_final_datasets()[OBJECTIVE]

# %% [markdown]
# ## Visualising the result
#
# We can take a look at where we queried the observer, both the original query points (crosses) and new query points (dots), and where they lie with respect to the contours of the Branin.

# %%
from util.plotting import plot_function_2d, plot_bo_points

arg_min_idx = tf.squeeze(tf.argmin(dataset.observations, axis=0))
query_points = dataset.query_points.numpy()
observations = dataset.observations.numpy()
_, ax = plot_function_2d(
    branin, search_space.lower, search_space.upper, grid_density=30, contour=True
)

plot_bo_points(query_points, ax[0, 0], num_initial_data_points, arg_min_idx)

# %% [markdown]
# We can also visualise the observations on a three-dimensional plot of the Branin. We'll add the contours of the mean and variance of the model's predictive distribution as translucent surfaces.

# %%
from util.plotting_plotly import plot_gp_plotly, add_bo_points_plotly

fig = plot_gp_plotly(
    result.try_get_final_models()[OBJECTIVE].model,
    search_space.lower,
    search_space.upper,
    grid_density=30
)
fig = add_bo_points_plotly(
    x=query_points[:, 0],
    y=query_points[:, 1],
    z=observations[:, 0],
    num_init=num_initial_data_points,
    idx_best=arg_min_idx,
    fig=fig,
    figrow=1,
    figcol=1,
)
fig.show()

# %% [markdown]
# ## LICENSE
#
# [Apache License 2.0](https://github.com/secondmind-labs/trieste/blob/develop/LICENSE)
