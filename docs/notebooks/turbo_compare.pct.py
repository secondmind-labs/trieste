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
# # DO NOT REVIEW. FOR TESTING ONLY. WILL BE REMOVED BEFORE MERGING.

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
from trieste.acquisition.rule import BatchTrustRegionBox, TURBOBox
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
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Mapping, Optional

from trieste import types
from trieste.acquisition.rule import (
    AcquisitionRule,
    AsynchronousOptimization,
    DiscreteThompsonSampling,
    EfficientGlobalOptimization,
    RandomSampling,
    ResultType,
)
from trieste.acquisition.utils import get_local_dataset
from trieste.data import Dataset
from trieste.models.interfaces import TrainableSupportsGetKernel
from trieste.space import Box, SearchSpace
from trieste.types import Tag


class TURBO(
    AcquisitionRule[
        types.State[Optional["TURBO.State"], TensorType], Box, TrainableSupportsGetKernel
    ]
):
    """Implements the TURBO algorithm as detailed in :cite:`eriksson2019scalable`."""

    @dataclass(frozen=True)
    class State:
        """The acquisition state for the :class:`TURBO` acquisition rule."""

        acquisition_space: Box
        """ The search space. """

        L: float
        """ Length of the trust region (before standardizing by model lengthscales) """

        failure_counter: int
        """ Number of consecutive failures (reset if we see a success). """

        success_counter: int
        """ Number of consecutive successes (reset if we see a failure).  """

        y_min: TensorType
        """ The minimum observed value. """

        def __deepcopy__(self, memo: dict[int, object]) -> TURBO.State:
            box_copy = copy.deepcopy(self.acquisition_space, memo)
            return TURBO.State(
                box_copy, self.L, self.failure_counter, self.success_counter, self.y_min
            )

    def __init__(
        self,
        search_space: SearchSpace,
        num_trust_regions: int = 1,
        rule: Optional[AcquisitionRule[ResultType, Box, TrainableSupportsGetKernel]] = None,
        L_min: Optional[float] = None,
        L_init: Optional[float] = None,
        L_max: Optional[float] = None,
        success_tolerance: int = 3,
        failure_tolerance: Optional[int] = None,
        local_models: Optional[Mapping[Tag, TrainableSupportsGetKernel]] = None,
    ):
        """
        Note that the optional parameters are set by a heuristic if not given by the user.

        :param search_space: The search space.
        :param num_trust_regions: Number of trust regions controlled by TURBO
        :param rule: rule used to select points from within the trust region, using the local model.
        :param L_min: Minimum allowed length of the trust region.
        :param L_init: Initial length of the trust region.
        :param L_max: Maximum allowed length of the trust region.
        :param success_tolerance: Number of consecutive successes before changing region size.
        :param failure tolerance: Number of consecutive failures before changing region size.
        :param local_models: Optional model to act as the local model. This will be refit using
            the data from each trust region. If no local_models are provided then we just
            copy the global model.
        """

        if not num_trust_regions > 0:
            raise ValueError(f"Num trust regions must be greater than 0, got {num_trust_regions}")

        if num_trust_regions > 1:
            raise NotImplementedError(
                f"TURBO does not yet support multiple trust regions, but got {num_trust_regions}"
            )

        # implement heuristic defaults for TURBO if not specified by user
        if rule is None:  # default to Thompson sampling with batches of size 1
            rule = DiscreteThompsonSampling(tf.minimum(100 * search_space.dimension, 5_000), 1)

        if failure_tolerance is None:
            if isinstance(
                rule,
                (
                    EfficientGlobalOptimization,
                    DiscreteThompsonSampling,
                    RandomSampling,
                    AsynchronousOptimization,
                ),
            ):
                failure_tolerance = math.ceil(search_space.dimension / rule._num_query_points)
            else:
                failure_tolerance == search_space.dimension
            assert isinstance(failure_tolerance, int)
        search_space_max_width = tf.reduce_max(search_space.upper - search_space.lower)
        if L_min is None:
            L_min = (0.5**7) * search_space_max_width
        if L_init is None:
            L_init = 0.8 * search_space_max_width
        if L_max is None:
            L_max = 1.6 * search_space_max_width

        if not success_tolerance > 0:
            raise ValueError(
                f"success tolerance must be an integer greater than 0, got {success_tolerance}"
            )
        if not failure_tolerance > 0:
            raise ValueError(
                f"success tolerance must be an integer greater than 0, got {failure_tolerance}"
            )

        if L_min <= 0:
            raise ValueError(f"L_min must be postive, got {L_min}")
        if L_init <= 0:
            raise ValueError(f"L_min must be postive, got {L_init}")
        if L_max <= 0:
            raise ValueError(f"L_min must be postive, got {L_max}")

        self._num_trust_regions = num_trust_regions
        self._L_min = L_min
        self._L_init = L_init
        self._L_max = L_max
        self._success_tolerance = success_tolerance
        self._failure_tolerance = failure_tolerance
        self._rule = rule
        self._local_models = local_models

    def __repr__(self) -> str:
        """"""
        return f"TURBO({self._num_trust_regions!r}, {self._rule})"

    def acquire(
        self,
        search_space: Box,
        models: Mapping[Tag, TrainableSupportsGetKernel],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> types.State[State | None, TensorType]:
        """
        Construct a local search space from ``search_space`` according the TURBO algorithm,
        and use that with the ``rule`` specified at :meth:`~TURBO.__init__` to find new
        query points. Return a function that constructs these points given a previous trust region
        state.

        If no ``state`` is specified (it is `None`), then we build the initial trust region.

        If a ``state`` is specified, and the new optimum improves over the previous optimum,
        the previous acquisition is considered successful.

        If ``success_tolerance`` previous consecutive acquisitions were successful then the search
        space is made larger. If  ``failure_tolerance`` consecutive acquisitions were unsuccessful
        then the search space is shrunk. If neither condition is triggered then the search space
        remains the same.

        **Note:** The acquisition search space will never extend beyond the boundary of the
        ``search_space``. For a local search, the actual search space will be the
        intersection of the trust region and ``search_space``.

        :param search_space: The local acquisition search space for *this step*.
        :param models: The model for each tag.
        :param datasets: The known observer query points and observations. Uses the data for key
            `OBJECTIVE` to calculate the new trust region.
        :return: A function that constructs the next acquisition state and the recommended query
            points from the previous acquisition state.
        :raise KeyError: If ``datasets`` does not contain the key `OBJECTIVE`.
        """
        if self._local_models is None:  # if user doesnt specifiy a local model
            self._local_models = copy.deepcopy(
                models
            )  # copy global model (will be fit locally later)

        if self._local_models.keys() != {OBJECTIVE}:
            raise ValueError(
                f"dict of models must contain the single key {OBJECTIVE}, got keys {models.keys()}"
            )

        if datasets is None or datasets.keys() != {OBJECTIVE}:
            raise ValueError(
                f"""datasets must be provided and contain the single key {OBJECTIVE}"""
            )

        dataset = datasets[OBJECTIVE]
        local_model = self._local_models[OBJECTIVE]
        global_lower = search_space.lower
        global_upper = search_space.upper

        y_min = tf.reduce_min(dataset.observations, axis=0)

        def state_func(
            state: TURBO.State | None,
        ) -> tuple[TURBO.State | None, TensorType]:
            if state is None:  # initialise first TR
                L, failure_counter, success_counter = self._L_init, 0, 0
            else:  # update TR
                step_is_success = y_min < state.y_min - 1e-10  # maybe make this stronger?
                failure_counter = (
                    0 if step_is_success else state.failure_counter + 1
                )  # update or reset counter
                success_counter = (
                    state.success_counter + 1 if step_is_success else 0
                )  # update or reset counter
                L = state.L
                if success_counter == self._success_tolerance:
                    L *= 2.0  # make region bigger
                    success_counter = 0
                elif failure_counter == self._failure_tolerance:
                    L *= 0.5  # make region smaller
                    failure_counter = 0

                L = tf.minimum(L, self._L_max)
                if L < self._L_min:  # if gets too small then start again
                    L, failure_counter, success_counter = self._L_init, 0, 0

            # build region with volume according to length L but stretched according to lengthscales
            xmin = dataset.query_points[tf.argmin(dataset.observations)[0], :]  # centre of region
            lengthscales = (
                local_model.get_kernel().lengthscales
            )  # stretch region according to model lengthscales
            tr_width = (
                lengthscales * L / tf.reduce_prod(lengthscales) ** (1.0 / global_lower.shape[-1])
            )  # keep volume fixed
            acquisition_space = Box(
                tf.reduce_max([global_lower, xmin - tr_width / 2.0], axis=0),
                tf.reduce_min([global_upper, xmin + tr_width / 2.0], axis=0),
            )

            # fit the local model using just data from the trust region
            local_dataset = get_local_dataset(acquisition_space, dataset)
            local_model.update(local_dataset)
            local_model.optimize(local_dataset)

            # use local model and local dataset to choose next query point(s)
            points = self._rule.acquire_single(acquisition_space, local_model, local_dataset)
            state_ = TURBO.State(acquisition_space, L, failure_counter, success_counter, y_min)

            return state_, points

        return state_func


# %%
np.random.seed(1793)
tf.random.set_seed(1793)

# %%
branin = ScaledBranin.objective
search_space = ScaledBranin.search_space

num_initial_data_points = 6
num_query_points = 1
num_steps = 15

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
acq_rule2 = TURBO(
    search_space,
    rule=EfficientGlobalOptimization(),
)
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
