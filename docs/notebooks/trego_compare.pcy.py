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

# %%
# from aim.ext.tensorboard_tracker import Run
from datetime import datetime

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.pyplot import cm

import trieste
from trieste.acquisition import ParallelContinuousThompsonSampling
from trieste.acquisition.optimizer import automatic_optimizer_selector
from trieste.acquisition.rule import BatchTrustRegionBox, TREGOBox, TrustRegion
from trieste.acquisition.utils import split_acquisition_function_calls
from trieste.ask_tell_optimization import AskTellOptimizer
from trieste.experimental.plotting import plot_regret
from trieste.experimental.plotting.plotting import create_grid
from trieste.logging import pyplot
from trieste.models.gpflow import GaussianProcessRegression, build_gpr
from trieste.objectives import Hartmann6, ScaledBranin
from trieste.types import TensorType

# %%
np.random.seed(179)
tf.random.set_seed(179)

# %%
branin = ScaledBranin.objective
search_space = ScaledBranin.search_space

num_initial_data_points = 6
num_query_points = 3
num_steps = 10

initial_query_points = search_space.sample(num_initial_data_points)
observer = trieste.objectives.utils.mk_observer(branin)
initial_data = observer(initial_query_points)

# %%
gpflow_model1 = build_gpr(
    initial_data,
    search_space,
    likelihood_variance=1e-4,
    trainable_likelihood=False,
)
model1 = GaussianProcessRegression(gpflow_model1)
acq_rule1 = BatchTrustRegionBox([TREGOBox(search_space)])  # type: ignore[var-annotated]
ask_tell1 = AskTellOptimizer(
    search_space, initial_data, model1, acquisition_rule=acq_rule1
)

# %%
gpflow_model2 = build_gpr(
    initial_data,
    search_space,
    likelihood_variance=1e-4,
    trainable_likelihood=False,
)
model2 = GaussianProcessRegression(gpflow_model2)
acq_rule2 = TrustRegion()
ask_tell2 = AskTellOptimizer(
    search_space, initial_data, model2, acquisition_rule=acq_rule2
)

# %%
for step in range(num_steps):
    print(f"step number {step}")
    new_points1 = ask_tell1.ask()
    new_data1 = observer(new_points1)
    ask_tell1.tell(new_data1)

    new_points2 = ask_tell2.ask()
    new_data2 = observer(new_points2)
    ask_tell2.tell(new_data2)

    assert ask_tell1._acquisition_state is not None
    assert ask_tell2._acquisition_state is not None
    np.testing.assert_array_almost_equal(new_points1, new_points2, decimal=4)
    np.testing.assert_array_almost_equal(
        new_data1.observations, new_data2.observations, decimal=4
    )
    np.testing.assert_equal(
        ask_tell1._acquisition_state.acquisition_space.get_subspace("0")._is_global,  # type: ignore
        ask_tell2._acquisition_state.is_global,  # type: ignore
    )

# %%
from trieste.experimental.plotting import plot_bo_points, plot_function_2d

bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

num_steps = 25
result = bo.optimize(
    num_steps, initial_data, model1, acq_rule1, track_state=False
)
dataset = result.try_get_final_dataset()


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

# %%
from trieste.experimental.plotting import plot_bo_points, plot_function_2d

bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

num_steps = 25
result = bo.optimize(
    num_steps, initial_data, model2, acq_rule2, track_state=False
)
dataset = result.try_get_final_dataset()


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
