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
# # Batch trust region Bayesian optimization

# %%
import numpy as np
import tensorflow as tf

np.random.seed(1793)
tf.random.set_seed(1793)

# %% [markdown]
# ## Define the problem and model
#
# You can use trust regions for Bayesian optimization in much the same way as we used EGO and EI in the [introduction notebook](expected_improvement.ipynb). Since the setup is much the same as in that tutorial, we'll skip over most of the detail.

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
# As usual, we'll use Gaussian process regression to model the function. Note that we set the likelihood variance to a small number because we are dealing with a noise-free problem.

# %%
from trieste.models.gpflow import GaussianProcessRegression, build_gpr

gpflow_model = build_gpr(initial_data, search_space, likelihood_variance=1e-7)
model = GaussianProcessRegression(gpflow_model)


# %% [markdown]
# ## Create the batch trust region acquisition rule
#
# We achieve Bayesian optimization with trust region by specifying `BatchTrustRegionBox` as the acquisition rule.
#
# This rule requires an initial number `num_query_points` of sub-spaces (or trust regions) to be provided and performs optimization in parallel across all these sub-spaces. Each region contributes one query point, resulting in each acquisition step collecting `num_query_points` points overall. As the optimization process continues, the bounds of these sub-spaces are dynamically updated.
#
# In addition, this rule requires the specification of a batch aquisition base-rule for performing optimization; for our example we use `EfficientGlobalOptimization` coupled with `ParallelContinuousThompsonSampling`.
#
# Note: the number of sub-spaces/regions must match the number of batch query points.

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
acq_rule = trieste.acquisition.rule.BatchTrustRegionBox(
    init_subspaces, base_rule
)

# %% [markdown]
# ## Run the optimization loop
#
# We can now run the Bayesian optimization loop by defining a `BayesianOptimizer` and calling its `optimize` method with the trust region rule. Once the optimization loop is complete, the optimizer will return `num_query_points` new query points for every step in the loop. With 5 steps, that's 25 points in total.

# %%
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

num_steps = 5
result = bo.optimize(num_steps, initial_data, model, acq_rule, track_state=True)
dataset = result.try_get_final_dataset()

# %% [markdown]
# ## Visualizing the result
#
# We can take a look at where we queried the observer, the original query points (crosses), new query points (dots) and the optimum point found (purple dot), and where they lie with respect to the contours of the Branin.

# %%
from trieste.experimental.plotting import plot_bo_points, plot_function_2d

arg_min_idx = tf.squeeze(tf.argmin(dataset.observations, axis=0))
query_points = dataset.query_points.numpy()
observations = dataset.observations.numpy()
_, ax = plot_function_2d(
    branin,
    search_space.lower,
    search_space.upper,
    grid_density=40,
    contour=True,
)

plot_bo_points(query_points, ax[0, 0], num_initial_data_points, arg_min_idx)

# %% [markdown]
# Here we visualize the observations on a three-dimensional plot of the Branin. We'll add the contours of the mean and variance of the model's predictive distribution as translucent surfaces.

# %%
from trieste.experimental.plotting import plot_model_predictions_plotly

fig = plot_model_predictions_plotly(
    result.try_get_final_model(),
    search_space.lower,
    search_space.upper,
)

from trieste.experimental.plotting import add_bo_points_plotly

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
# We can also visualize how each successive point compares with the current best by plotting regret.
# This plot shows the observations (crosses and dots), the current best (orange line), and the start of the optimization loop (blue line).

# %%
import matplotlib.pyplot as plt

from trieste.experimental.plotting import plot_regret

suboptimality = observations - Branin.minimum.numpy()

fig, ax = plt.subplots()
plot_regret(
    suboptimality, ax, num_init=num_initial_data_points, idx_best=arg_min_idx
)

ax.set_yscale("log")
ax.set_ylabel("Regret")
ax.set_xlabel("# evaluations")

# %% [markdown]
# Next we visualize the progress of the optimization by plotting the trust regions at each step. The trust regions are shown as translucent boxes, with the current optimum point in each region shown in matching color.

# %%
import base64
import io

import imageio
import IPython
from matplotlib.colors import rgb2hex
from matplotlib.patches import Rectangle
from matplotlib.pyplot import cm

colors = [
    rgb2hex(color) for color in cm.rainbow(np.linspace(0, 1, num_query_points))
]
frames = []

for step, hist in enumerate(result.history + [result.final_result.unwrap()]):
    state = hist.acquisition_state
    if state is None:
        continue

    # Plot branin contour.
    fig, ax = plot_function_2d(
        branin,
        search_space.lower,
        search_space.upper,
        grid_density=40,
        contour=True,
    )

    query_points = hist.dataset.query_points
    new_points_mask = np.zeros(query_points.shape[0], dtype=bool)
    new_points_mask[-num_query_points:] = True

    assert isinstance(state, trieste.acquisition.rule.BatchTrustRegionBox.State)
    acquisition_space = state.acquisition_space

    # Plot trust regions.
    for i, tag in enumerate(acquisition_space.subspace_tags):
        lb = acquisition_space.get_subspace(tag).lower
        ub = acquisition_space.get_subspace(tag).upper
        ax[0, 0].add_patch(
            Rectangle(
                (lb[0], lb[1]),
                ub[0] - lb[0],
                ub[1] - lb[1],
                facecolor=colors[i],
                edgecolor=colors[i],
                alpha=0.3,
            )
        )

    # Plot new query points, using failure mask to color them.
    plot_bo_points(
        query_points,
        ax[0, 0],
        num_initial_data_points,
        mask_fail=new_points_mask,
        c_pass="black",
        c_fail=colors,
    )

    fig.suptitle(f"step number {step}")
    fig.canvas.draw()
    size_pix = fig.get_size_inches() * fig.dpi
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
    frames.append(image.reshape(list(size_pix[::-1].astype(int)) + [3]))
    plt.close(fig)


# Create and show the GIF.
gif_file = io.BytesIO()
imageio.mimsave(gif_file, frames, format="gif", loop=0, duration=5000)
gif = IPython.display.HTML(
    '<img src="data:image/gif;base64,{0}"/>'.format(
        base64.b64encode(gif_file.getvalue()).decode()
    )
)
IPython.display.display(gif)

# %% [markdown]
# ## LICENSE
#
# [Apache License 2.0](https://github.com/secondmind-labs/trieste/blob/develop/LICENSE)
