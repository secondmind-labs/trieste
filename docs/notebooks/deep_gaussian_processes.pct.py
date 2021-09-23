# %% [markdown]
# # Using deep Gaussian processes with GPflux for Bayesian optimization.

# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import tensorflow as tf

tf.keras.backend.set_floatx("float64")

# %% [markdown]
# ## Describe the problem
# In this example, we look to find the minimum value of the two- and five-dimensional Michalewicz functions over the hypercubes $[0, pi]^2$/$[0, pi]^5$. We compare a two-layer DGP model with GPR, using Thompson sampling for both.
#
# The Michalewicz functions are highly non-stationary and have a global minimum that's hard to find, so DGPs might be more suitable than standard Gaussian processes.

# %%
import gpflow
from trieste.objectives import (
    michalewicz,
    MICHALEWICZ_2_MINIMUM,
    MICHALEWICZ_5_MINIMUM,
)
from trieste.objectives.utils import mk_observer
from util.plotting_plotly import plot_function_plotly
from trieste.space import Box
from math import pi

function = michalewicz
F_MINIMIZER = MICHALEWICZ_2_MINIMUM

search_space = Box([0, 0], [pi, pi])

fig = plot_function_plotly(
    function,
    search_space.lower,
    search_space.upper,
    grid_density=100
)
fig.update_layout(height=800, width=800)
fig.show()

# %% [markdown]
# ## Sample the observer over the search space
#
# We set up the observer as usual, using Sobol sampling to sample the initial points.

# %%
import trieste

observer = mk_observer(function)

num_initial_points = 20
num_acquisitions = 480

num_acq_per_loop = 10
num_loops = 48

initial_query_points = search_space.sample_sobol(num_initial_points)
initial_data = observer(initial_query_points)

# %% [markdown]
# ## Model the objective function
#
# The Bayesian optimization procedure estimates the next best points to query by using a probabilistic model of the objective. We'll use a two layer deep Gaussian process (DGP), built using GPflux. We also compare to a (shallow) GP.
#
# We note that the DGP model requires us to specify the number of inducing points, as we don't have the true posterior. We also have to use a stochastic optimizer, such as Adam. Fortunately, GPflux allows us to use the Keras `fit` method, which makes optimizing a lot easier!

# %%
from trieste.models.gpflux import GPfluxModelConfig, build_vanilla_deep_gp, build_gi_deep_gp
from gpflow.utilities import set_trainable


def build_dgp_model(data):
    variance = tf.math.reduce_variance(data.observations)

    dgp = build_gi_deep_gp(data.query_points, num_layers=2, num_inducing=100,
                           last_layer_variance=variance.numpy())
    # dgp = build_vanilla_deep_gp(data.query_points, num_layers=2, num_inducing=100)
    # dgp.f_layers[-1].kernel.kernel.variance.assign(variance)
    dgp.f_layers[-1].mean_function = gpflow.mean_functions.Constant()
    dgp.likelihood_layer.variance.assign(1e-3)
    # set_trainable(dgp.likelihood_layer, False)

    epochs = 200
    batch_size = 100

    optimizer = tf.optimizers.Adam(0.01)
    fit_args = {
        "batch_size": batch_size,
        "epochs": epochs,
        "verbose": 1,
    }

    return GPfluxModelConfig(**{
        "model": dgp,
        "model_args": {
            "fit_args": fit_args,
        },
        "optimizer": optimizer,
    })


# dgp_model = build_dgp_model(initial_data)

# %% [markdown]
# ## Run the optimization loop
#
# We can now run the Bayesian optimization loop by defining a `BayesianOptimizer` and calling its `optimize` method.
#
# The optimizer uses an acquisition rule to choose where in the search space to try on each optimization step. We'll start by using Thompson sampling.
#
# We'll run the optimizer for twenty steps. Note: this may take a while!
# %%
from trieste.acquisition.rule import DiscreteThompsonSampling
from util.plotting_plotly import plot_dgp_plotly, plot_gi_dgp_plotly
from util.plotting_plotly import add_bo_points_plotly

bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
acquisition_rule = DiscreteThompsonSampling(1000, 1)

dgp_dataset = initial_data

for loop in range(num_loops):
    dgp_model = build_dgp_model(dgp_dataset)
    dgp_result = bo.optimize(num_acq_per_loop, dgp_dataset, dgp_model,
                             acquisition_rule=acquisition_rule, track_state=False)
    dgp_dataset = dgp_result.try_get_final_dataset()

    dgp_query_points = dgp_dataset.query_points.numpy()
    dgp_observations = dgp_dataset.observations.numpy()
    dgp_arg_min_idx = tf.squeeze(tf.argmin(dgp_observations, axis=0))

    fig = plot_gi_dgp_plotly(
        dgp_result.try_get_final_model().model_gpflux,  # type: ignore
        search_space.lower,
        search_space.upper,
        grid_density=100,
        num_samples=200,
    )

    fig = add_bo_points_plotly(
        x=dgp_query_points[:, 0],
        y=dgp_query_points[:, 1],
        z=dgp_observations[:, 0],
        num_init=num_initial_points,
        idx_best=dgp_arg_min_idx,
        fig=fig,
        figrow=1,
        figcol=1,
    )
    fig.update_layout(height=800, width=800)
    fig.show()


# %% [markdown]
# ## Explore the results
#
# We can now get the best point found by the optimizer. Note this isn't necessarily the point that was last evaluated.

# %%
dgp_query_points = dgp_dataset.query_points.numpy()
dgp_observations = dgp_dataset.observations.numpy()

dgp_arg_min_idx = tf.squeeze(tf.argmin(dgp_observations, axis=0))

print(f"query point: {dgp_query_points[dgp_arg_min_idx, :]}")
print(f"observation: {dgp_observations[dgp_arg_min_idx, :]}")

# %% [markdown]
# We can visualise how the optimizer performed as a three-dimensional plot

# %%
from util.plotting_plotly import add_bo_points_plotly

fig = plot_function_plotly(function, search_space.lower, search_space.upper, grid_density=100)
fig.update_layout(height=800, width=800)

fig = add_bo_points_plotly(
    x=dgp_query_points[:, 0],
    y=dgp_query_points[:, 1],
    z=dgp_observations[:, 0],
    num_init=num_initial_points,
    idx_best=dgp_arg_min_idx,
    fig=fig,
)
fig.show()

import matplotlib.pyplot as plt
from util.plotting import plot_regret

dgp_suboptimality = dgp_observations - F_MINIMIZER.numpy()

# %% [markdown]
# We can visualise the model over the objective function by plotting the mean and 95% confidence intervals of its predictive distribution.

# %%
from util.plotting_plotly import plot_dgp_plotly, plot_gi_dgp_plotly

fig = plot_gi_dgp_plotly(
    dgp_result.try_get_final_model().model_gpflux,  # type: ignore
    search_space.lower,
    search_space.upper,
    grid_density=100,
    num_samples=50,
)

fig = add_bo_points_plotly(
    x=dgp_query_points[:, 0],
    y=dgp_query_points[:, 1],
    z=dgp_observations[:, 0],
    num_init=num_initial_points,
    idx_best=dgp_arg_min_idx,
    fig=fig,
    figrow=1,
    figcol=1,
)
fig.update_layout(height=800, width=800)
fig.show()

# %% [markdown]
# We now compare to a GP model with priors over the hyperparameters. We do not expect this to do as well because GP models cannot deal with non-stationary functions well.

# %%
import gpflow
import tensorflow_probability as tfp
from trieste.models.gpflow import GPflowModelConfig


def build_gp_model(data):
    variance = tf.math.reduce_variance(data.observations)
    kernel = gpflow.kernels.Matern52(variance=variance, lengthscales=[0.2]*data.query_points.shape[-1])
    prior_scale = tf.cast(1.0, dtype=tf.float64)
    kernel.variance.prior = tfp.distributions.LogNormal(tf.cast(-2.0, dtype=tf.float64), prior_scale)
    kernel.lengthscales.prior = tfp.distributions.LogNormal(tf.math.log(kernel.lengthscales), prior_scale)
    gpr = gpflow.models.GPR(data.astuple(), kernel, mean_function=gpflow.mean_functions.Constant(), noise_variance=1e-3)
    gpflow.set_trainable(gpr.likelihood, True)

    return GPflowModelConfig(**{
        "model": gpr,
        "model_args": {
            "num_kernel_samples": 100,
        },
        "optimizer": gpflow.optimizers.Scipy(),
        "optimizer_args": {
            "minimize_args": {"options": dict(maxiter=100)},
        },
    })


gp_model = build_gp_model(initial_data)

bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

result = bo.optimize(num_acquisitions, initial_data, gp_model, acquisition_rule=acquisition_rule,
                     track_state=False)
gp_dataset = result.try_get_final_dataset()

gp_query_points = gp_dataset.query_points.numpy()
gp_observations = gp_dataset.observations.numpy()

gp_arg_min_idx = tf.squeeze(tf.argmin(gp_observations, axis=0))

print(f"query point: {gp_query_points[gp_arg_min_idx, :]}")
print(f"observation: {gp_observations[gp_arg_min_idx, :]}")

gp_suboptimality = gp_observations - F_MINIMIZER.numpy()

from util.plotting_plotly import plot_gp_plotly

fig = plot_gp_plotly(
    result.try_get_final_model().model,  # type: ignore
    search_space.lower,
    search_space.upper,
    grid_density=100,
)

fig = add_bo_points_plotly(
    x=gp_query_points[:, 0],
    y=gp_query_points[:, 1],
    z=gp_observations[:, 0],
    num_init=num_initial_points,
    idx_best=gp_arg_min_idx,
    fig=fig,
    figrow=1,
    figcol=1,
)
fig.update_layout(height=800, width=800)
fig.show()

# %% [markdown]
# We plot the regret curves of the two models side-by-side.

# %%

_, ax = plt.subplots(1, 2)
plot_regret(dgp_suboptimality, ax[0], num_init=num_initial_points, idx_best=dgp_arg_min_idx)
plot_regret(gp_suboptimality, ax[1], num_init=num_initial_points, idx_best=gp_arg_min_idx)

ax[0].set_yscale("log")
ax[0].set_ylabel("Regret")
ax[0].set_ylim(0.00001, 2)
ax[0].set_xlabel("# evaluations")
ax[0].set_title("DGP")

ax[1].set_title("GP")
ax[1].set_yscale("log")
ax[1].set_ylim(0.00001, 2)
ax[1].set_xlabel("# evaluations")

plt.show()
plt.close()

# %% [markdown]
# We might also expect that the DGP model will do better on higher dimensional data. We explore this by testing a higher-dimensional version of the Michalewicz dataset.

# %%
from trieste.data import TensorType


def michalewicz_5(x: TensorType) -> TensorType:
    return michalewicz(x, 5)


function = michalewicz_5
F_MINIMIZER = MICHALEWICZ_5_MINIMUM

search_space = Box([0]*5, [pi]*5)

observer = mk_observer(function)

num_initial_points = 50
num_acquisitions = 50
initial_query_points = search_space.sample_sobol(num_initial_points)
initial_data = observer(initial_query_points)

dgp_model = build_dgp_model(initial_data)

bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
acquisition_rule = DiscreteThompsonSampling(1000, 1)

dgp_result = bo.optimize(num_acquisitions, initial_data, dgp_model,
                         acquisition_rule=acquisition_rule, track_state=False)
dgp_dataset = dgp_result.try_get_final_dataset()

dgp_query_points = dgp_dataset.query_points.numpy()
dgp_observations = dgp_dataset.observations.numpy()

dgp_arg_min_idx = tf.squeeze(tf.argmin(dgp_observations, axis=0))

print(f"query point: {dgp_query_points[dgp_arg_min_idx, :]}")
print(f"observation: {dgp_observations[dgp_arg_min_idx, :]}")

dgp_suboptimality = dgp_observations - F_MINIMIZER.numpy()

gp_model = build_gp_model(initial_data)

bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

result = bo.optimize(num_acquisitions, initial_data, gp_model, acquisition_rule=acquisition_rule,
                     track_state=False)
gp_dataset = result.try_get_final_dataset()

gp_query_points = gp_dataset.query_points.numpy()
gp_observations = gp_dataset.observations.numpy()

gp_arg_min_idx = tf.squeeze(tf.argmin(gp_observations, axis=0))

print(f"query point: {gp_query_points[gp_arg_min_idx, :]}")
print(f"observation: {gp_observations[gp_arg_min_idx, :]}")

gp_suboptimality = gp_observations - F_MINIMIZER.numpy()

_, ax = plt.subplots(1, 2)
plot_regret(dgp_suboptimality, ax[0], num_init=num_initial_points, idx_best=dgp_arg_min_idx)
plot_regret(gp_suboptimality, ax[1], num_init=num_initial_points, idx_best=gp_arg_min_idx)

ax[0].set_yscale("log")
ax[0].set_ylabel("Regret")
ax[0].set_ylim(1.5, 6)
ax[0].set_xlabel("# evaluations")
ax[0].set_title("DGP")

ax[1].set_title("GP")
ax[1].set_yscale("log")
ax[1].set_ylim(1.5, 6)
ax[1].set_xlabel("# evaluations")

# %% [markdown]
# While still far from the optimum, it is considerably better than the GP.

#

# ## LICENSE
#
# [Apache License 2.0](https://github.com/secondmind-labs/trieste/blob/develop/LICENSE)
