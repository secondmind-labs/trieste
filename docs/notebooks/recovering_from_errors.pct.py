# %% [markdown]
# # Recovering from errors during optimization

# %%
import numpy as np
import tensorflow as tf
import random

np.random.seed(1793)
tf.random.set_seed(1793)
random.seed(3)

# %% [markdown]
# Sometimes the Bayesian optimization process encounters an error from which we can recover, without the need to restart the run from the beginning. In this tutorial, we'll simulate such an error and show how to recover from it.
#
# We'll use a similar setup to the [EI notebook](expected_improvement.ipynb), but use an observer that intermittently breaks when evaluated, and needs manual attention to get running again. We can simulate fixing the observer with its `manual_fix` method.

# %%
import trieste
from trieste.objectives import branin


class FaultyBranin:
    def __init__(self):
        self._is_broken = False

    def manual_fix(self):
        self._is_broken = False

    def __call__(self, x):
        if random.random() < 0.05:
            self._is_broken = True

        if self._is_broken:
            raise Exception("Observer is broken")

        return trieste.data.Dataset(x, branin(x))


observer = FaultyBranin()

# %% [markdown]
# ## Set up the problem
# We'll use the same set up as before, except for the acquisition rule, where we'll use `TrustRegion`. `TrustRegion` is stateful, and we'll need to account for its state to recover, so using this rule gives the reader a more comprehensive overview of how to recover.

# %%
from trieste.models.gpflow import build_gpr, GaussianProcessRegression

search_space = trieste.space.Box(
    tf.cast([0.0, 0.0], tf.float64), tf.cast([1.0, 1.0], tf.float64)
)
initial_data = observer(search_space.sample(5))

gpr = build_gpr(initial_data, search_space)
model = GaussianProcessRegression(gpr)

acquisition_rule = trieste.acquisition.rule.TrustRegion()

# %% [markdown]
# ## Run the optimization loop
#
# In this tutorial we'll try to complete fifteen optimization loops, which, with the broken observer, may take more than one attempt. The optimizer returns an `OptimizationResult`, which is simply a container for both:
#
#   * the final result, which uses a `Result` type (not to be confused with `OptimizationResult`) to safely encapsulate the final data, models and acquisition state if the process completed successfully, or an error if one occurred
#   * the history of the successful optimization steps.
#
# We can access these with the `astuple` method.

# %%
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

num_steps = 15
result, history = bo.optimize(
    num_steps, initial_data, model, acquisition_rule, None
).astuple()

# %% [markdown]
# We can see from the logs that the optimization loop failed, and this can be sufficient to know what to do next if we're working in a notebook. However, sometimes our setup means we don't have access to the logs. We'll pretend from here that's the case.

# %% [markdown]
# ## Handling success
#
# We don't know if the optimization completed successfully or not, so we'll only try to access and plot the data if it was successful. We can find out if this was the case with `result`'s `is_ok` attribute. If it was successful, we know there is data in the `result`, which we can `unwrap` and view.

# %%
if result.is_ok:
    data = result.unwrap().dataset
    print("best observation: ", tf.reduce_min(data.observations))

# %% [markdown]
# ## Handling failure
#
# If on the other hand, the optimization didn't complete successfully, we can fix our observer, and try again. We can try again by using the data, model and acquisition state from the last successful step, which is the last element of the `history`. Recall that we only need to account for the acquisition state because we're using the stateful `TrustRegion` rule. For most rules, we don't need to account for this state.
#
# Note can view any `Result` by printing it. We'll do that here to see what exception was caught.

# %%
if result.is_err:
    print("result: ", result)

    observer.manual_fix()

    result, new_history = bo.optimize(
        15 - len(history),
        history[-1].dataset,
        history[-1].model,
        acquisition_rule,
        history[-1].acquisition_state,
    ).astuple()

    history.extend(new_history)

# %% [markdown]
# We can repeat this until we've spent our optimization budget, using a loop if appropriate. But here, we'll just plot the data if it exists, safely by using `result`'s `is_ok` attribute.

# %%
from util.plotting import plot_bo_points, plot_function_2d

if result.is_ok:
    data = result.unwrap().dataset
    arg_min_idx = tf.squeeze(tf.argmin(data.observations, axis=0))
    _, ax = plot_function_2d(
        branin, search_space.lower, search_space.upper, 30, contour=True
    )
    plot_bo_points(data.query_points.numpy(), ax[0, 0], 5, arg_min_idx)

# %% [markdown]
# ## Saving results to disk
#
# For convenience, tracked state is stored in memory by default. However, this can potentially result in Out of Memory errors and also makes it difficult to recover from intentional or unintentional Python process shutdowns. You can instead store the result on disk by passing in a `track_path` argument to `optimize`.
#
# **Note that trieste currently saves models using pickling, which is not portable and not secure. You should only try to load optimization results that you generated yourself on the same system (or a system with the same version libraries).**

# %%
result, history = bo.optimize(
    num_steps, initial_data, model, acquisition_rule, None, track_path="history"
).astuple()

# %% [markdown]
# The returned `history` records are now stored in files rather than in memory. Their constituents can be accessed just as before, which loads the content into memory only when required. The `result` is automatically loaded into memory, but is also saved to disk with the rest of the history.

# %%
print(history[-1])
print(history[-1].model)

# %% [markdown]
# It is also possible to reload the `OptimizationResult` in a new Python process:

# %%
trieste.bayesian_optimizer.OptimizationResult.from_path("history")

# %% [markdown]
# ## Out of memory errors
#
# Since Out Of Memory errors normally result in the Python process shutting down, saving tracked state to disk as described above is an important tool in recovering from them. One possible cause of memory errors is trying to evaluate an acquisition function over a large dataset, e.g. when initializing our gradient-based optimizers. To work around this, you can specify that evaluations of the acquisition function be split up: this splits them (on the first dimension) into batches of a given size, then stitches them back together. To do this, you need to provide an explicit split optimizer and specify a desired batch size.

# %%
from trieste.acquisition.optimizer import automatic_optimizer_selector
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.acquisition.utils import split_acquisition_function_calls

optimizer = split_acquisition_function_calls(
    automatic_optimizer_selector, split_size=10_000
)
query_rule = EfficientGlobalOptimization(optimizer=optimizer)
acquisition_rule = trieste.acquisition.rule.TrustRegion(rule=query_rule)

# %% [markdown]
# ## LICENSE
#
# [Apache License 2.0](https://github.com/secondmind-labs/trieste/blob/develop/LICENSE)
