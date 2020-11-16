# %% [markdown]
# Copyright 2020 The Trieste Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# %% [markdown]
# # Introduction

# %%
from dataclasses import astuple

import gpflow
from gpflow.utilities import print_summary, set_trainable
import numpy as np
import tensorflow as tf

import trieste
from trieste.bayesian_optimizer import OptimizationResult
from trieste.utils.objectives import branin, mk_observer
from trieste.acquisition.rule import OBJECTIVE

from util.plotting_plotly import (
    plot_function_plotly, plot_gp_plotly, add_bo_points_plotly
)
from util.plotting import plot_function_2d, plot_bo_points, plot_regret

# %%
gpflow.config.set_default_float(np.float64)
np.random.seed(1793)
tf.random.set_seed(1793)

# %% [markdown]
# ## Describe the problem
# In this example, we look to find the minimum value of the two-dimensional Branin function over the hypercube $[0, 1]^2$. We can plot contours of the Branin over this space.

# %%
mins = [0.0, 0.0]
maxs = [1.0, 1.0]

fig = plot_function_plotly(branin, mins, maxs, grid_density=20)
fig.update_layout(height=400, width=400)
fig.show()

# %% [markdown]
# ## Sample the observer over the search space
#
# Sometimes we don't have direct access to the objective function. We only have an observer that indirectly observes it. In _Trieste_, the observer outputs a number of datasets, each of which must be labelled so the optimization process knows which is which. In our case, we only have one dataset, the objective. We'll use _Trieste_'s default label for single-model setups, `OBJECTIVE`. We can convert a function with `branin`'s signature to a single-output observer using `mk_observer`.
#
# The optimization procedure will benefit from having some starting data from the objective function to base its search on. We sample five points from the search space and evaluate them on the observer. We can represent the search space using a `Box`.

# %%
observer = mk_observer(branin, OBJECTIVE)
lower_bound = tf.cast(mins, gpflow.default_float())
upper_bound = tf.cast(maxs, gpflow.default_float())
search_space = trieste.space.Box(lower_bound, upper_bound)

num_initial_points = 10
initial_query_points = search_space.sample(num_initial_points)
initial_data = observer(initial_query_points)

# %% [markdown]
# ## Model the objective function
#
# The Bayesian optimization procedure estimates the next best points to query by using a probabilistic model of the objective. We'll use Gaussian process regression for this, provided by GPflow. The model will need to be trained on each step as more points are evaluated, so we'll package it with GPflow's Scipy optimizer.
#
# Note we could leave it to the optimizer to build this model, but we'll want to inspect it later so we'll build it ourselves.
#
# Just like the data output by the observer, the optimization process assumes multiple models, so we'll need to label the model in the same way.

# %%
def build_model(data):
    variance = tf.math.reduce_variance(data.observations)
    kernel = gpflow.kernels.Matern52(variance=variance, lengthscales=0.2 * np.ones(2,))
    gpr = gpflow.models.GPR(astuple(data), kernel, noise_variance=1e-5)
    set_trainable(gpr.likelihood, False)

    return {OBJECTIVE: trieste.models.create_model_interface({
        "model": gpr,
        "optimizer": gpflow.optimizers.Scipy(),
        "optimizer_args": {"options": dict(maxiter=100)},
    })}

model = build_model(initial_data[OBJECTIVE])

# %% [markdown]
# # Try an alternative acquisition function

# %% [markdown]
# By default, Trieste uses Expected Improvement (EI) as its acqusition function when performing Bayesian optimization. However, many alternative acqusition functions have been developed. One such alternative is Max-value Entropy Search (MES), which approximates the distribution of current estimate of the global minimum and tries to decrease its entropy with each optimization step.

# %% [markdown]
# We plot these two acquisition functions across our search space. Areas with high acquisition function scores (i.e bright regions) are those rated as promising locations for the next evaluation of our objective function. We see that EI wishes to continue exploring the search space, whereas MES wants to focus resources on evaluating a specific region.

# %%
# ei = trieste.acquisition.ExpectedImprovement()
# ei_acq_function = ei.using(OBJECTIVE).prepare_acquisition_function(initial_data, model)
# mcei = trieste.acquisition.MonteCarloExpectedImprovement(
#     eps_shape=[1000, 1, 1]  # [S, B, L]
# )
# mcei_acq_function = mcei.using(OBJECTIVE).prepare_acquisition_function(initial_data, model)
#
# fig, ax = plot_function_2d(mcei_acq_function, mins, maxs, grid_density=40, contour=True)
# plot_bo_points(
#     initial_data['OBJECTIVE'].query_points.numpy(),
#     ax=ax[0, 0],
# )
# fig.suptitle("MC-EI Acquisition Function")
#
# fig, ax = plot_function_2d(ei_acq_function, mins, maxs, grid_density=40, contour=True)
# plot_bo_points(
#     initial_data['OBJECTIVE'].query_points.numpy(),
#     ax=ax[0, 0],
# )
# fig.suptitle("Expected Improvement Acquisition Function")

# %% [markdown]
# To compare the performance of the optimization achieved by these two different acquisition functions, we re-run the above BO loop using MES.

# %% [markdown]
# We re-initialize the model and define a new acquisiton rule.

# %%
# model = build_model(initial_data[OBJECTIVE])
# acq_rule = trieste.acquisition.rule.EfficientGlobalOptimization(mcei.using(OBJECTIVE))

# %% [markdown]
# All that remains is to run the whole BO loop for 15 steps.
#
# # %%
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
#
# result = bo.optimize(2, initial_data, model, acquisition_rule=acq_rule)
#
# if result.error is not None: raise result.error
#
# dataset = result.datasets[OBJECTIVE]
#
# arg_min_idx = tf.squeeze(tf.argmin(dataset.observations, axis=0))
# fig, ax = plot_function_2d(branin, mins, maxs, grid_density=40, contour=True)
#
# plot_bo_points(
#     dataset.query_points.numpy(),
#     ax=ax[0, 0],
#     num_init=len(initial_query_points),
#     idx_best=arg_min_idx,
# )

batch_mcei = trieste.acquisition.MonteCarloExpectedImprovement(
    eps_shape=[200, 2, 1]  # [S, B, L]
)
batch_acq_rule = trieste.acquisition.rule.BatchAcquisitionRule(num_query_points=2,
                                                               builder=batch_mcei.using(OBJECTIVE))

batch_result = bo.optimize(10, initial_data, model, acquisition_rule=batch_acq_rule)

if batch_result.error is not None: raise batch_result.error

dataset = batch_result.datasets[OBJECTIVE]

arg_min_idx = tf.squeeze(tf.argmin(dataset.observations, axis=0))
fig, ax = plot_function_2d(branin, mins, maxs, grid_density=40, contour=True)

plot_bo_points(
    dataset.query_points.numpy(),
    ax=ax[0, 0],
    num_init=len(initial_query_points),
    idx_best=arg_min_idx,
)