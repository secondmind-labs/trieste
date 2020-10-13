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
# # Optimization with Max-value Entropy Search

# %%
from dataclasses import astuple

import numpy as np
import tensorflow as tf
import gpflow

import trieste
from trieste.utils.objectives import branin, mk_observer
from trieste.acquisition.rule import OBJECTIVE

from util.plotting_plotly import plot_gp_plotly, add_bo_points_plotly
from util.plotting import plot_function_2d, plot_bo_points

# %%
gpflow.config.set_default_float(tf.float64)
tf.random.set_seed(1793)

# %% [markdown]
# ## Define the problem and model
#
# You can use Max-value Entropy Search (MES) for Bayesian optimization in much the same way as we used EGO and EI
# in the tutorial _Introduction_. Since the setup is much the same is in that tutorial, we'll skip
# over most of the detail.
#
# We'll use a continuous bounded search space, and evaluate the observer at the point maximising the MES acquisition function.
#
# The MES acquisition function approximates the distribution of the value at the global
# minimum and tries to decrease its entropy. 
#
# See this paper for more details:
#     Z. Wang, S. Jegelka
#     Max-value Entropy Search for Efficient Bayesian Optimization
#     ICML 2017

# %%
lower_bound = tf.constant([0.0, 0.0], gpflow.default_float())
upper_bound = tf.constant([1.0, 1.0], gpflow.default_float())
search_space = trieste.space.Box(lower_bound, upper_bound)

num_initial_data_points = 10
initial_query_points = search_space.sample(num_initial_data_points)
observer = mk_observer(branin, OBJECTIVE)
initial_data = observer(initial_query_points)

# %% [markdown]
# We'll use Gaussian process regression to model the function.

# %%
observations = initial_data[OBJECTIVE].observations
kernel = gpflow.kernels.Matern52(tf.math.reduce_variance(observations), 0.2 * np.ones(2,))
gpr = gpflow.models.GPR(astuple(initial_data[OBJECTIVE]), kernel, noise_variance=1e-5)
gpflow.set_trainable(gpr.likelihood, False)

model_config = {OBJECTIVE: {
    "model": gpr,
    "optimizer": gpflow.optimizers.Scipy(),
    "optimizer_args": {"options": dict(maxiter=100)},
}}

# %% [markdown]
# ## Create the Max-value Entropy Search acquisition function
#
# We achieve Bayesian optimization with MES by specifying `MaxValueEntropySearch` as the
# acquisition rule. Just like the `EfficientGlobalOptimization` acquisition rule, `MaxValueEntropySearch` returns the point that maxmimises its acuisition function.
#
# Our implementation of MES is controlled by two parameters: "num_samples" and "grid_size". "num_samples" controls how many mote-carlo samples we use to calculate entropy reductions. As we only approximate a 1-d integral, "num_samples" does not need to be large or be increased for problems with large search space dimensions. We recomend values between 5-15. "grid_size" controls the coarseness of the grid used to approximate the distribution of our max value and so must increase with d. We recommend 1000*d. Note that as the grid must only be calculated once per BO step, the choice of "grid_size" does not have a large impact on computation time.
#

# %%
acq_rule = trieste.acquisition.rule.MaxValueEntropySearch(search_space,num_samples = 10, grid_size = 5000)


# %% [markdown]
# ## Run the optimization loop
#
# All that remains is to run the whole BO loop for 25 steps.

# %%
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

result = bo.optimize(25, initial_data, model_config, acq_rule)

if result.error is not None: raise result.error

dataset = result.datasets[OBJECTIVE]
mes_best = np.minimum.accumulate(result.datasets[OBJECTIVE].observations)[num_initial_data_points:]

# %% [markdown]
# ## Visualising the result
#
# We can take a look at where we queried the observer, both the original query points (crosses) and
# new query points (dots), and where they lie with respect to the contours of the Branin.

# %%
observations = initial_data[OBJECTIVE].observations
kernel = gpflow.kernels.Matern52(tf.math.reduce_variance(observations), 0.2 * np.ones(2,))
gpr = gpflow.models.GPR(astuple(initial_data[OBJECTIVE]), kernel, noise_variance=1e-5)
gpflow.set_trainable(gpr.likelihood, False)

model_config = {OBJECTIVE: {
    "model": gpr,
    "optimizer": gpflow.optimizers.Scipy(),
    "optimizer_args": {"options": dict(maxiter=100)},
}}

bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

result = bo.optimize(25, initial_data, model_config)

if result.error is not None: raise result.error
ego_best = np.minimum.accumulate(result.datasets[OBJECTIVE].observations)[num_initial_data_points:]

# %%
arg_min_idx = tf.squeeze(tf.argmin(dataset.observations, axis=0))
query_points = dataset.query_points.numpy()
observations = dataset.observations.numpy()
_, ax = plot_function_2d(
    branin, lower_bound.numpy(), upper_bound.numpy(), grid_density=30, contour=True
)

plot_bo_points(query_points, ax=ax[0, 0], num_init=num_initial_data_points, idx_best=arg_min_idx)

# %% [markdown]
# We can also visualise the observations on a three-dimensional plot of the Branin. We'll add
# the contours of the mean and variance of the model's predictive distribution as translucent
# surfaces.

# %%
fig = plot_gp_plotly(gpr, lower_bound.numpy(), upper_bound.numpy(), grid_density=30)
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
