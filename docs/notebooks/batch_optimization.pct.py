# %% [markdown]
# # Batch Bayesian Optimization with Batch Expected Improvement and Local Penalization

# %% [markdown]
# Sometimes it is practically convenient to query several points at a time. This notebook demonstrates two ways to perfom batch Bayesian optimization with `trieste`.

# %%
import numpy as np
import tensorflow as tf
from util.plotting import create_grid, plot_acq_function_2d
from util.plotting_plotly import plot_function_plotly
import matplotlib.pyplot as plt
import trieste

np.random.seed(42)
tf.random.set_seed(42)

# %% [markdown]
# ## Describe the problem
#
# In this example, we consider the same problem presented in our `expected_improvement` notebook, i.e. seeking the minimizer of the two-dimensional Branin function.
#
# We begin our optimization after collecting five function evaluations from random locations in the search space.

# %%
from trieste.utils.objectives import branin, mk_observer, BRANIN_MINIMUM
from trieste.space import Box

observer = mk_observer(branin)
search_space = Box([0, 0], [1, 1])

num_initial_points = 5
initial_query_points = search_space.sample(num_initial_points)
initial_data = observer(initial_query_points)

# %% [markdown]
# ## Surrogate model
# Just like in purely sequential optimization, we fit a surrogate Gaussian process model to the initial data.

# %%
import gpflow
from trieste.models import create_model
from trieste.utils import map_values


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


model_spec = build_model(initial_data)
model = create_model(model_spec)

# %% [markdown]
# ## Batch acquisition functions.
# To perform batch BO, we must define a batch acquisition function. Two popular batch acquisition functions supported in Trieste are `BatchMonteCarloExpectedImprovement` and the `LocalPenalizationAcquisitionFunction`. Although both of these acquisition functions recommend batches of diverse query points, the batches are chosen in very different ways. `BatchMonteCarloExpectedImprovement` jointly allocates the batch of points as those with the largest expected improvement over our current best solution. In contrast, the `LocalPenalizationAcquisitionFunction` greedily builds the batch, sequentially adding the maximizers of the standard (non-batch) `ExpectedImprovement` function penalized around the current pending batch points. In practice, `BatchMonteCarloExpectedImprovement` can be expected to have superior performance for small batches (`batch_size`<10) but scales poorly for larger batches.
#
# Note that both of these acquisition functions have controllable parameters. In particular, `BatchMonteCarloExpectedImprovement` is computed using a Monte-Carlo method (so it requires a `sample_size`), but uses a reparametrisation trick to make it deterministic. The `LocalPenalizationAcquisitionFunction` has parameters controlling the degree of penalization that must be estimated from a random sample of `num_samples` model predictions.

# %% [markdown]
# First, we collect the batch of ten points recommended by `BatchMonteCarloExpectedImprovement` ...

# %%
from trieste.acquisition import BatchMonteCarloExpectedImprovement
from trieste.acquisition.rule import EfficientGlobalOptimization

batch_ei_acq = BatchMonteCarloExpectedImprovement(sample_size=1000)
batch_ei_acq_rule = EfficientGlobalOptimization(  # type: ignore
    num_query_points=10, builder=batch_ei_acq)
points_chosen_by_batch_ei, _ = batch_ei_acq_rule.acquire_single(search_space, initial_data, model)

# %% [markdown]
# and then do the same with `LocalPenalizationAcquisitionFunction`.

# %%
from trieste.acquisition import LocalPenalizationAcquisitionFunction

local_penalization_acq = LocalPenalizationAcquisitionFunction(search_space, num_samples=1000)
local_penalization_acq_rule = EfficientGlobalOptimization(  # type: ignore
    num_query_points=10, builder=local_penalization_acq)
points_chosen_by_local_penalization, _ = local_penalization_acq_rule.acquire_single(
    search_space, initial_data, model)

# %% [markdown]
# We can now visualize the batch of 10 points chosen by each of these methods overlayed on the standard `ExpectedImprovement` acquisition function. `BatchMonteCarloExpectedImprovement` chooses a more diverse set of points, whereas the `LocalPenalizationAcquisitionFunction` focuses evaluations in the most promising areas of the space.

# %%
from trieste.acquisition import ExpectedImprovement

# plot standard EI acquisition function
ei = ExpectedImprovement()
ei_acq_function = ei.prepare_acquisition_function(initial_data, model)
plot_acq_function_2d(ei_acq_function, [0, 0], [1, 1], contour=True, grid_density=100)

plt.scatter(
    points_chosen_by_batch_ei[:, 0],
    points_chosen_by_batch_ei[:, 1],
    color="red",
    lw=5,
    label="Batch-EI",
    marker="*",
    zorder=1,
)
plt.scatter(
    points_chosen_by_local_penalization[:, 0],
    points_chosen_by_local_penalization[:, 1],
    color="black",
    lw=10,
    label="Local \nPenalization",
    marker="+",
)

plt.legend(bbox_to_anchor=(1.2, 1), loc="upper left")
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
cbar = plt.colorbar()
cbar.set_label("EI", rotation=270)

# %% [markdown]
# ## Run the batch optimization loop
# We can now run a batch Bayesian optimization loop by defining a `BayesianOptimizer` with one of our batch acquisition functions.
#
# We reuse the same ` EfficientGlobalOptimization` rule as in the purely sequential case, however we pass in one of batch acquisition functions and set `num_query_points`>1.
#
# We'll run each method for ten steps for batches of three points.

# %% [markdown]
# First we run ten steps of `BatchMonteCarloExpectedImprovement`

# %%
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

batch_ei_rule = EfficientGlobalOptimization(  # type: ignore
    num_query_points=3, builder=batch_ei_acq
)
qei_result = bo.optimize(10, initial_data, model, acquisition_rule=batch_ei_rule)

# %% [markdown]
# and then repeat the same optimization with `LocalPenalizationAcquisitionFunction`.

# %%
local_penalization_rule = EfficientGlobalOptimization(  # type: ignore
    num_query_points=3, builder=local_penalization_acq
)
local_penalization_result = bo.optimize(
    10, initial_data, model, acquisition_rule=local_penalization_rule
)

# %% [markdown]
# We can visualize the performance of each of these methods by plotting the trajectory of the regret (suboptimality) of the best observed solution as the optimization progresses. We denote this trajectory with the orange line, the start of the optimization loop with the blue line and the best overall point as a purple dot.
#
# For this particular problem (and random seed), we see that the`LocalPenalizationAcquisitionFunction` provides more effective batch optimization, finding a solution with a magnitude smaller regret than `BatchMonteCarloExpectedImprovement` under the same optimization budget.

# %%
from util.plotting import plot_regret

qei_observations = qei_result.try_get_final_dataset().observations - BRANIN_MINIMUM
qei_min_idx = tf.squeeze(tf.argmin(qei_observations, axis=0))
local_penalization_observations = (
    local_penalization_result.try_get_final_dataset().observations - BRANIN_MINIMUM
)
local_penalization_min_idx = tf.squeeze(tf.argmin(local_penalization_observations, axis=0))

_, ax = plt.subplots(1, 2)
plot_regret(qei_observations.numpy(), ax[0], num_init=5, idx_best=qei_min_idx)
ax[0].set_yscale("log")
ax[0].set_ylabel("Regret")
ax[0].set_ylim(0.01, 100)
ax[0].set_xlabel("# evaluations")
ax[0].set_title("Batch-EI")

plot_regret(local_penalization_observations.numpy(), ax[1], num_init=5, idx_best=local_penalization_min_idx)
ax[1].set_yscale("log")
ax[1].set_xlabel("# evaluations")
ax[1].set_ylim(0.01, 100)
ax[1].set_title("Local Penalization")

# %% [markdown]
# ## LICENSE
#
# [Apache License 2.0](https://github.com/secondmind-labs/trieste/blob/develop/LICENSE)
