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
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Silence TensorFlow warnings.

# from aim.ext.tensorboard_tracker import Run
from datetime import datetime
from typing import Mapping

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.pyplot import cm

import trieste
from trieste.acquisition import (
    DiscreteThompsonSampling,
    EfficientGlobalOptimization,
    ParallelContinuousThompsonSampling,
)
from trieste.acquisition.optimizer import automatic_optimizer_selector
from trieste.acquisition.rule import TURBO, BatchTrustRegionBox, TURBOBox
from trieste.acquisition.utils import copy_to_local_models
from trieste.ask_tell_optimization import AskTellOptimizer
from trieste.experimental.plotting import plot_regret
from trieste.experimental.plotting.plotting import create_grid
from trieste.logging import pyplot
from trieste.models.gpflow import GaussianProcessRegression, build_gpr
from trieste.objectives import Hartmann6, ScaledBranin
from trieste.objectives.utils import mk_batch_observer
from trieste.observer import OBJECTIVE
from trieste.types import TensorType
from trieste.utils.misc import LocalizedTag

# %%
np.random.seed(8934)
tf.random.set_seed(8934)

# %%
branin = ScaledBranin.objective
search_space = ScaledBranin.search_space

num_initial_data_points = 6
num_query_points = 1
num_steps = 20

initial_query_points = search_space.sample(num_initial_data_points)
observer = trieste.objectives.utils.mk_observer(branin)
batch_observer = mk_batch_observer(observer)
initial_data = observer(initial_query_points)

# %%
gpflow_model1 = build_gpr(
    initial_data,
    search_space,
    likelihood_variance=1e-4,
    trainable_likelihood=False,
)
model1 = copy_to_local_models(
    GaussianProcessRegression(gpflow_model1), num_query_points
)
acq_rule1 = BatchTrustRegionBox(  # type: ignore[var-annotated]
    TURBOBox(search_space),
    rule=EfficientGlobalOptimization()
    # rule=DiscreteThompsonSampling(tf.minimum(100 * search_space.dimension, 5_000), 1)
    # rule=DiscreteThompsonSampling(500, num_query_points)
)
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
# acq_rule2 = TURBO(search_space, rule=DiscreteThompsonSampling(500, num_query_points))
acq_rule2 = TURBO(search_space, rule=EfficientGlobalOptimization())
ask_tell2 = AskTellOptimizer(
    search_space, initial_data, model2, acquisition_rule=acq_rule2
)

# %%
np.testing.assert_array_almost_equal(
    ask_tell1.models[LocalizedTag(OBJECTIVE, 0)].get_kernel().lengthscales,
    ask_tell2.models[OBJECTIVE].get_kernel().lengthscales,
)

# %%
from tests.util.misc import empty_dataset

_ = ask_tell1.ask()
ask_tell1.tell(
    {
        OBJECTIVE: empty_dataset([2], [1]),
        LocalizedTag(OBJECTIVE, 0): empty_dataset([2], [1]),
    }
)

# %%
for step in range(num_steps):
    print(f"step number {step+1}")

    lengthscales1 = tf.constant(
        ask_tell1.models[LocalizedTag(OBJECTIVE, 0)].get_kernel().lengthscales
    )
    y_min1 = ask_tell1._acquisition_state.acquisition_space.get_subspace("0").y_min  # type: ignore
    L1 = ask_tell1._acquisition_state.acquisition_space.get_subspace("0").L  # type: ignore
    success_counter1 = ask_tell1._acquisition_state.acquisition_space.get_subspace("0").success_counter  # type: ignore
    failure_counter1 = ask_tell1._acquisition_state.acquisition_space.get_subspace("0").failure_counter  # type: ignore
    lower1 = ask_tell1._acquisition_state.acquisition_space.get_subspace("0").lower  # type: ignore
    upper1 = ask_tell1._acquisition_state.acquisition_space.get_subspace("0").upper  # type: ignore

    new_points1 = ask_tell1.ask()
    new_data1 = batch_observer(new_points1)
    ask_tell1.tell(new_data1)

    new_points2 = ask_tell2.ask()
    new_data2 = observer(new_points2)
    ask_tell2.tell(new_data2)

    assert isinstance(new_data1, Mapping)
    new_points1 = tf.squeeze(new_points1, axis=0)

    np.testing.assert_array_almost_equal(
        lengthscales1,
        ask_tell2._acquisition_rule._local_models[OBJECTIVE].get_kernel().lengthscales,  # type: ignore
        decimal=4,
    )

    assert ask_tell1._acquisition_state is not None
    assert ask_tell2._acquisition_state is not None
    np.testing.assert_array_almost_equal(
        y_min1,
        ask_tell2._acquisition_state.y_min,  # type: ignore
        decimal=4,
    )
    np.testing.assert_equal(
        L1,
        ask_tell2._acquisition_state.L,  # type: ignore
    )
    np.testing.assert_equal(
        success_counter1,
        ask_tell2._acquisition_state.success_counter,  # type: ignore
    )
    np.testing.assert_equal(
        failure_counter1,
        ask_tell2._acquisition_state.failure_counter,  # type: ignore
    )
    np.testing.assert_array_almost_equal(
        lower1,
        ask_tell2._acquisition_state.acquisition_space.lower,  # type: ignore
        decimal=4,
    )
    np.testing.assert_array_almost_equal(
        upper1,
        ask_tell2._acquisition_state.acquisition_space.upper,  # type: ignore
        decimal=4,
    )
    np.testing.assert_array_almost_equal(new_points1, new_points2, decimal=4)
    np.testing.assert_array_almost_equal(
        new_data1[OBJECTIVE].observations, new_data2.observations, decimal=4
    )

# %%
from trieste.experimental.plotting import plot_bo_points, plot_function_2d


def plot_final_result(
    _dataset: trieste.data.Dataset, num_init_points=num_initial_data_points
) -> None:
    arg_min_idx = tf.squeeze(tf.argmin(_dataset.observations, axis=0))
    query_points = _dataset.query_points.numpy()
    _, ax = plot_function_2d(
        branin,
        search_space.lower,
        search_space.upper,
        grid_density=40,
        contour=True,
    )

    plot_bo_points(query_points, ax[0, 0], num_init_points, arg_min_idx)


# %%
import base64

import IPython
import matplotlib.pyplot as plt

from trieste.experimental.plotting import (
    convert_figure_to_frame,
    convert_frames_to_gif,
    plot_trust_region_history_2d,
)


def plot_history(
    result: trieste.bayesian_optimizer.OptimizationResult,
    num_init_points=num_initial_data_points,
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
            num_init=num_init_points,
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


# %%
bo1 = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

num_steps = 25
result1 = bo1.optimize(
    num_steps, initial_data, model1, acq_rule1, track_state=True
)
dataset1 = result1.try_get_final_dataset()

# %%
plot_final_result(dataset1)

# %%
plot_history(result1)

# %%
bo2 = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

num_steps = 25
result2 = bo2.optimize(
    num_steps, initial_data, model2, acq_rule2, track_state=True
)
dataset2 = result2.try_get_final_dataset()

# %%
plot_final_result(dataset2)

# %%
plot_history(result2)
