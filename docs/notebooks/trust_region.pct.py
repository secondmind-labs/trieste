# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: .venv_310
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Trust region Bayesian optimization
#
# This notebook guides you through practical examples of trust region Bayesian optimization,
# illustrating algorithms like TREGO <cite data-cite="diouane2022trego"/> and TuRBO
# <cite data-cite="eriksson2019scalable"/> that could be beneficial for optimizing high-dimensional
# spaces. Trust region approaches adaptively constrain the search space to "trustworthy" regions where
# the model's predictions are deemed reliable. Trieste provides a flexible framework for implementing
# custom algorithms by encapsulating the behavior of rules and regions into separate abstract classes,
# `BatchTrustRegion` and `UpdatableTrustRegion` respectively.

# %%
import numpy as np
import tensorflow as tf

np.random.seed(1793)
tf.random.set_seed(1793)

# %% [markdown]
# ## Define the problem and model
#
# We can use trust regions for Bayesian optimization in much the same way as we used EGO and EI in
# the [introduction notebook](expected_improvement.ipynb). Since the setup is very similar to
# that tutorial, we'll skip over most of the detail.

# %%
import trieste
from trieste.objectives import Branin

branin = Branin.objective
search_space = Branin.search_space

num_initial_data_points = 10
initial_query_points = search_space.sample(num_initial_data_points)
observer = trieste.objectives.utils.mk_observer(branin)
initial_data = observer(initial_query_points)

# %% [markdown]
# As usual, we'll use Gaussian process regression to model the function. Note that we set the
# likelihood variance to a small number because we are dealing with a noise-free problem.

# %%
from trieste.models.gpflow import GaussianProcessRegression, build_gpr


def build_model():
    gpflow_model = build_gpr(
        initial_data, search_space, likelihood_variance=1e-7
    )
    return GaussianProcessRegression(gpflow_model)


# %% [markdown]
# ## Trust region `TREGO` acquisition rule
#
# First we demonstrate how to run Bayesian optimization with the `TREGO` algorithm, which alternates
# between regular EGO steps and local steps within one trust region (see
# <cite data-cite="diouane2022trego"/>).
#
# ### Create `TREGO` rule and run optimization loop
#
# We can run the Bayesian optimization loop by defining a `BayesianOptimizer` and calling its
# `optimize` method with the trust region rule. Once the optimization loop is complete, the
# optimizer will return one new query point for every step in the loop; that's 5 points in total.
#
# The trust region rule is created by instantiating the concrete `BatchTrustRegionBox` class. This
# is a "meta" rule that manages the acquisition from potentially multiple regions by applying a
# base-rule to each region. The default base-rule is `EfficientGlobalOptimization`, but a different
# base-rule can be provided as an argument to `BatchTrustRegionBox`. Here we explicitly set it to
# make usage clear.
#
# The regions themselves are implemented as separate classes. In this example we use a single
# instance of the `TREGOBox` class, which is responsible for managing initialization and update of
# the one region of the `TREGO` algorithm.

# %%
trego_acq_rule = trieste.acquisition.rule.BatchTrustRegionBox(
    trieste.acquisition.rule.TREGOBox(search_space),
    rule=trieste.acquisition.rule.EfficientGlobalOptimization(),
)
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

num_steps = 5
result = bo.optimize(
    num_steps, initial_data, build_model(), trego_acq_rule, track_state=True
)
dataset = result.try_get_final_dataset()

# %% [markdown]
# ### Visualizing `TREGO` results
#
# Let's take a look at where we queried the observer, the original query points (crosses), new
# query points (dots) and the optimum point found (purple dot), and where they lie with respect to
# the contours of the Branin.

# %%
from trieste.experimental.plotting import plot_bo_points, plot_function_2d


def plot_final_result(_dataset: trieste.data.Dataset) -> None:
    arg_min_idx = tf.squeeze(tf.argmin(_dataset.observations, axis=0))
    query_points = _dataset.query_points.numpy()
    _, ax = plot_function_2d(
        branin,
        search_space.lower,
        search_space.upper,
        grid_density=40,
        contour=True,
    )

    plot_bo_points(query_points, ax[0, 0], num_initial_data_points, arg_min_idx)


plot_final_result(dataset)

# %% [markdown]
# We can also visualize the progress of the optimization by plotting the acquisition space at each
# step. This space is either the full search space or the trust region, depending on the step, and
# is shown as a translucent box. The new query points per region are plotted in matching color.
#
# Note there is only one trust region in this plot, however the rules in the following sections will
# show multiple trust regions.

# %%
import base64
from typing import Optional

import IPython
import matplotlib.pyplot as plt

from trieste.experimental.plotting import (
    convert_figure_to_frame,
    convert_frames_to_gif,
    plot_trust_region_history_2d,
)


def plot_history(
    result: trieste.bayesian_optimizer.OptimizationResult,
    num_query_points: Optional[int] = None,
) -> None:
    frames = []
    for step, hist in enumerate(
        result.history + [result.final_result.unwrap()]
    ):
        fig, _ = plot_trust_region_history_2d(
            branin,
            search_space.lower,
            search_space.upper,
            hist,
            num_query_points=num_query_points,
            num_init=num_initial_data_points,
        )

        if fig is not None:
            fig.suptitle(f"step number {step}")
            frames.append(convert_figure_to_frame(fig))
            plt.close(fig)

    gif_file = convert_frames_to_gif(frames)
    gif = IPython.display.HTML(
        '<img src="data:image/gif;base64,{0}"/>'.format(
            base64.b64encode(gif_file.getvalue()).decode()
        )
    )
    IPython.display.display(gif)


plot_history(result)

# %% [markdown]
# ## Batch trust region rule
#
# Next we demonstrate how to run Bayesian optimization with the batch trust region rule.
#
# ### Create the batch trust region acquisition rule
#
# We achieve Bayesian optimization with trust regions by specifying `BatchTrustRegionBox` as the
# acquisition rule.
#
# This rule needs an initial number `num_query_points` of sub-spaces (or trust regions) to be
# provided and performs optimization in parallel across all these sub-spaces. Each region
# contributes one query point, resulting in each acquisition step collecting `num_query_points`
# points overall. As the optimization process continues, the bounds of these sub-spaces are
# dynamically updated. In this example, we create 5 `SingleObjectiveTrustRegionBox` regions. This
# class encapsulates the behavior of a trust region in a single sub-space; being responsible for
# maintaining its own state, initializing it, and updating it after each step.
#
# In addition, `BatchTrustRegionBox` is a "meta" rule that requires the specification of a
# batch aquisition base-rule for performing optimization; for our example we use
# `EfficientGlobalOptimization` coupled with the `ParallelContinuousThompsonSampling` acquisition
# function.
#
# Note: in this example the number of sub-spaces/regions is equal to the number of batch query
# points in the base-rule. This results in each region contributing one query point to the overall
# batch. However, it is possible to generate multiple query points from each region by setting
# `num_query_points` to be a multiple `Q` of the number of regions. In this case, each region will
# contribute `Q` query points to the overall batch.

# %%
num_query_points = 5

init_subspaces = [
    trieste.acquisition.rule.SingleObjectiveTrustRegionBox(search_space)
    for _ in range(num_query_points)
]
base_rule = trieste.acquisition.rule.EfficientGlobalOptimization(  # type: ignore[var-annotated]
    builder=trieste.acquisition.ParallelContinuousThompsonSampling(),
    num_query_points=num_query_points,
)
batch_acq_rule = trieste.acquisition.rule.BatchTrustRegionBox(
    init_subspaces, base_rule
)

# %% [markdown]
# ### Run the optimization loop
#
# We run the Bayesian optimization loop as before by defining a `BayesianOptimizer` and calling its
# `optimize` method with the trust region rule. Once the optimization loop is complete, the
# optimizer will return `num_query_points` new query points for every step in the loop. With
# 5 steps, that's 25 points in total.

# %%
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

num_steps = 5
result = bo.optimize(
    num_steps, initial_data, build_model(), batch_acq_rule, track_state=True
)
dataset = result.try_get_final_dataset()

# %% [markdown]
# ### Visualizing batch trust region results
#
# We visualize the results as before.

# %%
plot_final_result(dataset)

# %%
plot_history(result)

# %% [markdown]
# ## Trust region `TuRBO` acquisition rule
#
# Finally, we show how to run Bayesian optimization with the `TuRBO` algorithm. This is a
# trust region algorithm that uses local models and datasets to approximate the objective function
# within their respective trust regions (see <cite data-cite="eriksson2019scalable"/>).
#
# ### Create `TuRBO` rule and run optimization loop
#
# As before, this meta-rule requires the specification of an aquisition base-rule for performing
# optimization within the trust regions; for our example we use the `DiscreteThompsonSampling` rule.
#
# We create 2 `TuRBO` trust regions and associated local models by initially copying the global model
# (using `copy_to_local_models`).
#
# The optimizer will return `num_query_points` new query points for each region in every step of the
# loop. With 5 steps and 2 regions, that's 30 points in total.

# %%
num_regions = 2
num_query_points = 3

turbo_subspaces = [
    trieste.acquisition.rule.TURBOBox(search_space) for _ in range(num_regions)
]
dts_rule = trieste.acquisition.rule.DiscreteThompsonSampling(
    500, num_query_points
)
turbo_acq_rule = trieste.acquisition.rule.BatchTrustRegionBox(
    turbo_subspaces, dts_rule
)

bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

num_steps = 5
result = bo.optimize(
    num_steps,
    {trieste.observer.OBJECTIVE: initial_data},
    trieste.acquisition.utils.copy_to_local_models(build_model(), num_regions),
    turbo_acq_rule,
    track_state=True,
)
dataset = result.try_get_final_dataset()

# %% [markdown]
# ### Visualizing `TuRBO` results
#
# We display the results as earlier.

# %%
plot_final_result(dataset)

# %%
plot_history(result, num_regions * num_query_points)

# %% [markdown]
# ## LICENSE
#
# [Apache License 2.0](https://github.com/secondmind-labs/trieste/blob/develop/LICENSE)
