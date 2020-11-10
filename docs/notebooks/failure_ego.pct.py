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
# # EGO with a failure region

# %%
from dataclasses import astuple

import gpflow
from gpflow import set_trainable
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import trieste
from trieste.utils import objectives
from trieste.models.model_interfaces import VariationalGaussianProcess
from gpflow.models import VGP
from gpflow.optimizers import NaturalGradient

from util.plotting_plotly import (
    plot_function_plotly, plot_gp_plotly, add_bo_points_plotly
)
from util.plotting import plot_gp_2d, plot_function_2d, plot_bo_points

# %%
gpflow.config.set_default_jitter(1e-5)
gpflow.config.set_default_float(np.float64)
np.random.seed(1234)
tf.random.set_seed(1234)

# %% [markdown]
# ## The problem
#
# This notebook is similar to the _Introduction_ notebook, where we look to find the minimum value of the two-dimensional Branin function over the hypercube $[0, 1]^2$. But here, we constrain the problem, by adding an area to the search space in which the objective fails to evaluate.
#
# We represent this setup with a function `masked_branin` that produces null values when evaluated in the disk with center $(0.5, 0.4)$ and radius $0.3$. It's important to remember that while _we_ know where this _failure region_ is, this function is a black box from the optimizer's point of view: the optimizer must learn it.

# %%
def masked_branin(x):
    mask_nan = np.sqrt((x[:, 0] - 0.5) ** 2 + (x[:, 1] - .4) ** 2) < 0.3
    y = np.array(objectives.branin(x))
    y[mask_nan] = np.nan
    return tf.convert_to_tensor(y.reshape(-1, 1), x.dtype)

# %% [markdown]
# As mentioned, we'll search over the hypercube $[0, 1]^2$ ...

# %%
mins = [0.0, 0.0]
maxs = [1.0, 1.0]

lower_bound = tf.constant(mins, gpflow.default_float())
upper_bound = tf.constant(maxs, gpflow.default_float())
search_space = trieste.space.Box(lower_bound, upper_bound)

# %% [markdown]
# ... where the `masked_branin` now looks as follows. The white area in the centre shows the failure
# region.

# %%
fig = plot_function_plotly(masked_branin, mins, maxs, grid_density=70)
fig.update_layout(height=400, width=400)
fig.show()

# %% [markdown]
# ## Define the data sets
#
# We'll work with two data sets
#
#   - one containing only those query_points and observations where the observations are finite.
#     We'll label this with `OBJECTIVE`.
#   - the other containing all the query points, but whose observations indicate if evaluating the
#     observer failed at that point, using `1` if the evaluation failed, else `0`. We'll label this
#     with `FAILURE`.
#
# Let's define an observer that outputs the data in these formats.

# %%
OBJECTIVE = "OBJECTIVE"
FAILURE = "FAILURE"

def observer(x):
    y = masked_branin(x)
    mask = np.isfinite(y).reshape(-1)
    return {
        OBJECTIVE: trieste.data.Dataset(x[mask], y[mask]),
        FAILURE: trieste.data.Dataset(x, tf.cast(np.isfinite(y), tf.float64))
    }

# %% [markdown]
# We can evaluate the observer at points sampled from the search space.

# %%
num_init_points = 15
initial_data = observer(search_space.sample(num_init_points))


# %% [markdown]
# ## Model the data
#
# We'll model the data on the objective with a regression model, and the data on which points failed with a classification model. The regression model will be a `GaussianProcessRegression` wrapping a GPflow `GPR`, and the classification model a `VariationalGaussianProcess` wrapping a GPflow `VGP` with Bernoulli likelihood.

# %%
def create_regression_model(data):
    variance = tf.math.reduce_variance(data.observations)
    kernel = gpflow.kernels.Matern52(variance=variance, lengthscales=0.2 * np.ones(2, ))
    gpr = gpflow.models.GPR(astuple(data), kernel, noise_variance=1e-5)
    set_trainable(gpr.likelihood, False)
    return gpr


def create_classification_model(data):
    kernel = gpflow.kernels.SquaredExponential(variance=100., lengthscales=0.2 * np.ones(2, ))
    likelihood = gpflow.likelihoods.Bernoulli()
    vgp = gpflow.models.VGP(astuple(data), kernel, likelihood)
    set_trainable(vgp.kernel.variance, False)
    return vgp


regression_model = create_regression_model(initial_data[OBJECTIVE])
classification_model = create_classification_model(initial_data[FAILURE])

# %% [markdown]
# ## Create a custom optimize method
# The new `NatGradTrainedVGP` class has a custom `optimize` method that alternates between Adam steps to optimize the lengthscales and NatGrad steps to optimize the variational parameters:

# %%
class NatGradTrainedVGP(VariationalGaussianProcess):
    def optimize(self):
        set_trainable(self.model.q_mu, False)
        set_trainable(self.model.q_sqrt, False)
        variational_params = [(self.model.q_mu, self.model.q_sqrt)]
        adam_opt = tf.optimizers.Adam(1e-3)
        natgrad_opt = NaturalGradient(gamma=0.1)

        for step in range(100):
            natgrad_opt.minimize(self.model.training_loss, var_list=variational_params)
            adam_opt.minimize(self.model.training_loss, var_list=self.model.trainable_variables)

# %% [markdown]
# We'll train the GPR model with an L-BFGS-based optimizer, and the GPC model with the custom algorithm above.

# %%
models = {
    OBJECTIVE: {
        "model": regression_model,
        "optimizer": gpflow.optimizers.Scipy(),
        "optimizer_args": {"options": dict(maxiter=100)},
    },
    FAILURE: NatGradTrainedVGP(classification_model)
}

# %% [markdown]
# ## Create a custom acquisition function
#
# We'll need a custom acquisition function for this problem. This function is the product of the expected improvement for the objective data and the predictive mean for the failure data. We can specify which data and model to use in each acquisition function builder with the `OBJECTIVE` and `FAILURE` labels. We'll optimize the function using EfficientGlobalOptimization.

# %%
class ProbabilityOfValidity(trieste.acquisition.SingleModelAcquisitionBuilder):
    def prepare_acquisition_function(self, dataset, model):
        return lambda at: trieste.acquisition.lower_confidence_bound(model, 0.0, at)

ei = trieste.acquisition.ExpectedImprovement()
pov = ProbabilityOfValidity()
acq_fn = trieste.acquisition.Product(ei.using(OBJECTIVE), pov.using(FAILURE))
rule = trieste.acquisition.rule.EfficientGlobalOptimization(acq_fn)

# %% [markdown]
# ## Run the optimizer
#
# Now, we run the Bayesian optimization loop for twenty steps, and print the location of the query point corresponding to the minimum observation.

# %%
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

result = bo.optimize(20, initial_data, models, acquisition_rule=rule)

if result.error is not None: raise result.error

final_data = result.datasets

arg_min_idx = tf.squeeze(tf.argmin(final_data[OBJECTIVE].observations, axis=0))
print(f"query point: {final_data[OBJECTIVE].query_points[arg_min_idx, :]}")

# %% [markdown]
# We can visualise where the optimizer queried on a contour plot of the Branin with the failure region. The minimum observation can be seen along the bottom axis towards the right, outside of the failure region.

# %%
mask_fail = final_data[FAILURE].observations.numpy().flatten().astype(int) == 0
fig, ax = plot_function_2d(masked_branin, mins, maxs, grid_density=50, contour=True)
plot_bo_points(
    final_data[FAILURE].query_points.numpy(),
    ax=ax[0, 0],
    num_init=num_init_points,
    mask_fail=mask_fail,
)
plt.show()

# %% [markdown]
# We can also plot the mean and variance of the predictive distribution over the search space, first for the objective data and model ...

# %%
arg_min_idx = tf.squeeze(tf.argmin(final_data[OBJECTIVE].observations, axis=0))

fig = plot_gp_plotly(regression_model, mins, maxs, grid_density=50)
fig = add_bo_points_plotly(
    x=final_data[OBJECTIVE].query_points[:, 0].numpy(),
    y=final_data[OBJECTIVE].query_points[:, 1].numpy(),
    z=final_data[OBJECTIVE].observations.numpy().flatten(),
    num_init=num_init_points,
    idx_best=arg_min_idx,
    fig=fig,
    figrow=1,
    figcol=1,
)

fig.show()

# %% [markdown]
# ... and then for the failure data and model

# %%
fig, ax = plot_gp_2d(
    classification_model,
    mins,
    maxs,
    grid_density=50,
    contour=True,
    figsize=(12, 5),
    predict_y=True,
)

plot_bo_points(
    final_data[FAILURE].query_points.numpy(),
    num_init=num_init_points,
    ax=ax[0, 0],
    mask_fail=mask_fail,
)

plot_bo_points(
    final_data[FAILURE].query_points.numpy(),
    num_init=num_init_points,
    ax=ax[0, 1],
    mask_fail=mask_fail,
)

plt.show()
