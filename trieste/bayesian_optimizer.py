# Copyright 2021 The Trieste Contributors
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

"""
This module contains the :class:`BayesianOptimizer` class, used to perform Bayesian optimization.
"""

from __future__ import annotations

import copy
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Dict, Generic, TypeVar, cast, overload

import numpy as np
import tensorflow as tf
from absl import logging

from .acquisition.rule import AcquisitionRule, EfficientGlobalOptimization
from .data import Dataset
from .logging import get_tensorboard_writer, set_step_number
from .models import ModelSpec, TrainableProbabilisticModel, create_model
from .models.config import ModelConfigType
from .observer import OBJECTIVE, Observer
from .space import SearchSpace
from .types import State, TensorType
from .utils import Err, Ok, Result, Timer, map_values

StateType = TypeVar("StateType")
""" Unbound type variable. """

SearchSpaceType = TypeVar("SearchSpaceType", bound=SearchSpace)
""" Type variable bound to :class:`SearchSpace`. """

TrainableProbabilisticModelType = TypeVar(
    "TrainableProbabilisticModelType", bound=TrainableProbabilisticModel, contravariant=True
)
""" Contravariant type variable bound to :class:`TrainableProbabilisticModel`. """


@dataclass(frozen=True)
class Record(Generic[StateType]):
    """Container to record the state of each step of the optimization process."""

    datasets: Mapping[str, Dataset]
    """ The known data from the observer. """

    models: Mapping[str, TrainableProbabilisticModel]
    """ The models over the :attr:`datasets`. """

    acquisition_state: StateType | None
    """ The acquisition state. """

    @property
    def dataset(self) -> Dataset:
        """The dataset when there is just one dataset."""
        if len(self.datasets) == 1:
            return next(iter(self.datasets.values()))
        else:
            raise ValueError(f"Expected a single dataset, found {len(self.datasets)}")

    @property
    def model(self) -> TrainableProbabilisticModel:
        """The model when there is just one dataset."""
        if len(self.models) == 1:
            return next(iter(self.models.values()))
        else:
            raise ValueError(f"Expected a single dataset, found {len(self.datasets)}")


# this should be a generic NamedTuple, but mypy doesn't support them
#  https://github.com/python/mypy/issues/685
@dataclass(frozen=True)
class OptimizationResult(Generic[StateType]):
    """The final result, and the historical data of the optimization process."""

    final_result: Result[Record[StateType]]
    """
    The final result of the optimization process. This contains either a :class:`Record` or an
    exception.
    """

    history: list[Record[StateType]]
    r"""
    The history of the :class:`Record`\ s from each step of the optimization process. These
    :class:`Record`\ s are created at the *start* of each loop, and as such will never include the
    :attr:`final_result`.
    """

    def astuple(self) -> tuple[Result[Record[StateType]], list[Record[StateType]]]:
        """
        **Note:** In contrast to the standard library function :func:`dataclasses.astuple`, this
        method does *not* deepcopy instance attributes.

        :return: The :attr:`final_result` and :attr:`history` as a 2-tuple.
        """
        return self.final_result, self.history

    def try_get_final_datasets(self) -> Mapping[str, Dataset]:
        """
        Convenience method to attempt to get the final data.

        :return: The final data, if the optimization completed successfully.
        :raise Exception: If an exception occurred during optimization.
        """
        return self.final_result.unwrap().datasets

    def try_get_final_dataset(self) -> Dataset:
        """
        Convenience method to attempt to get the final data for a single dataset run.

        :return: The final data, if the optimization completed successfully.
        :raise Exception: If an exception occurred during optimization.
        :raise ValueError: If the optimization was not a single dataset run.
        """
        datasets = self.try_get_final_datasets()
        if len(datasets) == 1:
            return next(iter(datasets.values()))
        else:
            raise ValueError(f"Expected a single dataset, found {len(datasets)}")

    def try_get_final_models(self) -> Mapping[str, TrainableProbabilisticModel]:
        """
        Convenience method to attempt to get the final models.

        :return: The final models, if the optimization completed successfully.
        :raise Exception: If an exception occurred during optimization.
        """
        return self.final_result.unwrap().models

    def try_get_final_model(self) -> TrainableProbabilisticModel:
        """
        Convenience method to attempt to get the final model for a single model run.

        :return: The final model, if the optimization completed successfully.
        :raise Exception: If an exception occurred during optimization.
        :raise ValueError: If the optimization was not a single model run.
        """
        models = self.try_get_final_models()
        if len(models) == 1:
            return next(iter(models.values()))
        else:
            raise ValueError(f"Expected single model, found {len(models)}")


class BayesianOptimizer(Generic[SearchSpaceType]):
    """
    This class performs Bayesian optimization, the data-efficient optimization of an expensive
    black-box *objective function* over some *search space*. Since we may not have access to the
    objective function itself, we speak instead of an *observer* that observes it.
    """

    def __init__(self, observer: Observer, search_space: SearchSpaceType):
        """
        :param observer: The observer of the objective function.
        :param search_space: The space over which to search. Must be a
            :class:`~trieste.space.SearchSpace`.
        """
        self._observer = observer
        self._search_space = search_space

    def __repr__(self) -> str:
        """"""
        return f"BayesianOptimizer({self._observer!r}, {self._search_space!r})"

    @overload
    def optimize(
        self,
        num_steps: int,
        datasets: Mapping[str, Dataset],
        model_specs: Mapping[str, ModelSpec],
        *,
        track_state: bool = True,
        fit_initial_model: bool = True,
    ) -> OptimizationResult[None]:
        ...

    @overload
    def optimize(
        self,
        num_steps: int,
        datasets: Mapping[str, Dataset],
        model_specs: Mapping[str, TrainableProbabilisticModelType],
        acquisition_rule: AcquisitionRule[
            TensorType, SearchSpaceType, TrainableProbabilisticModelType
        ],
        *,
        track_state: bool = True,
        fit_initial_model: bool = True,
        # this should really be OptimizationResult[None], but tf.Tensor is untyped so the type
        # checker can't differentiate between TensorType and State[S | None, TensorType], and
        # the return types clash. object is close enough to None that object will do.
    ) -> OptimizationResult[object]:
        ...

    @overload
    def optimize(
        self,
        num_steps: int,
        datasets: Mapping[str, Dataset],
        # there's no way to statically check config-based models
        model_specs: Mapping[str, ModelConfigType],
        acquisition_rule: AcquisitionRule[
            TensorType, SearchSpaceType, TrainableProbabilisticModelType
        ],
        *,
        track_state: bool = True,
        fit_initial_model: bool = True,
        # this should really be OptimizationResult[None], but tf.Tensor is untyped so the type
        # checker can't differentiate between TensorType and State[S | None, TensorType], and
        # the return types clash. object is close enough to None that object will do.
    ) -> OptimizationResult[object]:
        ...

    @overload
    def optimize(
        self,
        num_steps: int,
        datasets: Mapping[str, Dataset],
        model_specs: Mapping[str, TrainableProbabilisticModelType],
        acquisition_rule: AcquisitionRule[
            State[StateType | None, TensorType], SearchSpaceType, TrainableProbabilisticModelType
        ],
        acquisition_state: StateType | None = None,
        *,
        track_state: bool = True,
        fit_initial_model: bool = True,
    ) -> OptimizationResult[StateType]:
        ...

    @overload
    def optimize(
        self,
        num_steps: int,
        datasets: Mapping[str, Dataset],
        # there's no way to statically check config-based models
        model_specs: Mapping[str, ModelConfigType],
        acquisition_rule: AcquisitionRule[
            State[StateType | None, TensorType], SearchSpaceType, TrainableProbabilisticModelType
        ],
        acquisition_state: StateType | None = None,
        *,
        track_state: bool = True,
        fit_initial_model: bool = True,
    ) -> OptimizationResult[StateType]:
        ...

    @overload
    def optimize(
        self,
        num_steps: int,
        datasets: Dataset,
        model_specs: ModelSpec,
        *,
        track_state: bool = True,
        fit_initial_model: bool = True,
    ) -> OptimizationResult[None]:
        ...

    @overload
    def optimize(
        self,
        num_steps: int,
        datasets: Dataset,
        model_specs: TrainableProbabilisticModelType,
        acquisition_rule: AcquisitionRule[
            TensorType, SearchSpaceType, TrainableProbabilisticModelType
        ],
        *,
        track_state: bool = True,
        fit_initial_model: bool = True,
    ) -> OptimizationResult[object]:
        ...

    @overload
    def optimize(
        self,
        num_steps: int,
        datasets: Dataset,
        model_specs: ModelConfigType,
        acquisition_rule: AcquisitionRule[
            TensorType, SearchSpaceType, TrainableProbabilisticModelType
        ],
        *,
        track_state: bool = True,
        fit_initial_model: bool = True,
    ) -> OptimizationResult[object]:
        ...

    @overload
    def optimize(
        self,
        num_steps: int,
        datasets: Dataset,
        model_specs: TrainableProbabilisticModelType,
        acquisition_rule: AcquisitionRule[
            State[StateType | None, TensorType], SearchSpaceType, TrainableProbabilisticModelType
        ],
        acquisition_state: StateType | None = None,
        *,
        track_state: bool = True,
        fit_initial_model: bool = True,
    ) -> OptimizationResult[StateType]:
        ...

    @overload
    def optimize(
        self,
        num_steps: int,
        datasets: Dataset,
        model_specs: ModelConfigType,
        acquisition_rule: AcquisitionRule[
            State[StateType | None, TensorType], SearchSpaceType, TrainableProbabilisticModelType
        ],
        acquisition_state: StateType | None = None,
        *,
        track_state: bool = True,
        fit_initial_model: bool = True,
    ) -> OptimizationResult[StateType]:
        ...

    def optimize(
        self,
        num_steps: int,
        datasets: Mapping[str, Dataset] | Dataset,
        model_specs: Mapping[str, TrainableProbabilisticModelType]
        | Mapping[str, ModelSpec]
        | TrainableProbabilisticModelType
        | ModelConfigType,
        acquisition_rule: AcquisitionRule[
            TensorType | State[StateType | None, TensorType],
            SearchSpaceType,
            TrainableProbabilisticModelType,
        ]
        | None = None,
        acquisition_state: StateType | None = None,
        *,
        track_state: bool = True,
        fit_initial_model: bool = True,
    ) -> OptimizationResult[StateType] | OptimizationResult[None]:
        """
        Attempt to find the minimizer of the ``observer`` in the ``search_space`` (both specified at
        :meth:`__init__`). This is the central implementation of the Bayesian optimization loop.

        For each step in ``num_steps``, this method:
            - Finds the next points with which to query the ``observer`` using the
              ``acquisition_rule``'s :meth:`acquire` method, passing it the ``search_space``,
              ``datasets``, models built from the ``model_specs``, and current acquisition state.
            - Queries the ``observer`` *once* at those points.
            - Updates the datasets and models with the data from the ``observer``.

        If any errors are raised during the optimization loop, this method will catch and return
        them instead, along with the history of the optimization process, and print a message (using
        `absl` at level `logging.ERROR`).

        **Note:** While the :class:`~trieste.models.TrainableProbabilisticModel` interface implies
        mutable models, it is *not* guaranteed that the model passed to :meth:`optimize` will
        be updated during the optimization process. For example, if ``track_state`` is `True`, a
        copied model will be used on each optimization step. Use the models in the return value for
        reliable access to the updated models.

        **Type hints:**
            - The ``acquisition_rule`` must use the same type of
              :class:`~trieste.space.SearchSpace` as specified in :meth:`__init__`.
            - The ``acquisition_state`` must be of the type expected by the ``acquisition_rule``.
              Any acquisition state in the optimization result will also be of this type.

        :param num_steps: The number of optimization steps to run.
        :param datasets: The known observer query points and observations for each tag.
        :param model_specs: The model to use for each :class:`~trieste.data.Dataset` in
            ``datasets``.
        :param acquisition_rule: The acquisition rule, which defines how to search for a new point
            on each optimization step. Defaults to
            :class:`~trieste.acquisition.rule.EfficientGlobalOptimization` with default
            arguments. Note that if the default is used, this implies the tags must be
            `OBJECTIVE`, the search space can be any :class:`~trieste.space.SearchSpace`, and the
            acquisition state returned in the :class:`OptimizationResult` will be `None`.
        :param acquisition_state: The acquisition state to use on the first optimization step.
            This argument allows the caller to restore the optimization process from an existing
            :class:`Record`.
        :param track_state: If `True`, this method saves the optimization state at the start of each
            step. Models and acquisition state are copied using `copy.deepcopy`.
        :param fit_initial_model: If `False`, this method assumes that the initial models have
            already been optimized on the datasets and so do not require optimization before the
            first optimization step.
        :return: An :class:`OptimizationResult`. The :attr:`final_result` element contains either
            the final optimization data, models and acquisition state, or, if an exception was
            raised while executing the optimization loop, it contains the exception raised. In
            either case, the :attr:`history` element is the history of the data, models and
            acquisition state at the *start* of each optimization step (up to and including any step
            that fails to complete). The history will never include the final optimization result.
        :raise ValueError: If any of the following are true:

            - ``num_steps`` is negative.
            - the keys in ``datasets`` and ``model_specs`` do not match
            - ``datasets`` or ``model_specs`` are empty
            - the default `acquisition_rule` is used and the tags are not `OBJECTIVE`.
        """
        if isinstance(datasets, Dataset):
            datasets = {OBJECTIVE: datasets}
            model_specs = {OBJECTIVE: model_specs}

        # reassure the type checker that everything is tagged
        datasets = cast(Dict[str, Dataset], datasets)
        model_specs = cast(Dict[str, ModelSpec], model_specs)

        if num_steps < 0:
            raise ValueError(f"num_steps must be at least 0, got {num_steps}")

        if datasets.keys() != model_specs.keys():
            raise ValueError(
                f"datasets and model_specs should contain the same keys. Got {datasets.keys()} and"
                f" {model_specs.keys()} respectively."
            )

        if not datasets:
            raise ValueError("dicts of datasets and model_specs must be populated.")

        if acquisition_rule is None:
            if datasets.keys() != {OBJECTIVE}:
                raise ValueError(
                    f"Default acquisition rule EfficientGlobalOptimization requires tag"
                    f" {OBJECTIVE!r}, got keys {datasets.keys()}"
                )

            acquisition_rule = EfficientGlobalOptimization[
                SearchSpaceType, TrainableProbabilisticModelType
            ]()

        # note that this cast is justified for explicit models but not for models created
        # from config, which can't be statically type checked; those will fail at runtime instead
        models = cast(
            Dict[str, TrainableProbabilisticModelType], map_values(create_model, model_specs)
        )

        history: list[Record[StateType]] = []

        for step in range(num_steps):
            set_step_number(step)
            try:

                if track_state:
                    models_copy = copy.deepcopy(models)
                    acquisition_state_copy = copy.deepcopy(acquisition_state)
                    history.append(Record(datasets, models_copy, acquisition_state_copy))

                with Timer() as total_step_wallclock_timer:
                    if step == 0 and fit_initial_model:
                        with Timer() as initial_model_fitting_timer:
                            for tag, model in models.items():
                                dataset = datasets[tag]
                                model.update(dataset)
                                model.optimize(dataset)

                    with Timer() as query_point_generation_timer:
                        points_or_stateful = acquisition_rule.acquire(
                            self._search_space, models, datasets=datasets
                        )

                    if callable(points_or_stateful):
                        acquisition_state, query_points = points_or_stateful(acquisition_state)
                    else:
                        query_points = points_or_stateful

                    observer_output = self._observer(query_points)

                    tagged_output = (
                        observer_output
                        if isinstance(observer_output, Mapping)
                        else {OBJECTIVE: observer_output}
                    )

                    datasets = {tag: datasets[tag] + tagged_output[tag] for tag in tagged_output}

                    with Timer() as model_fitting_timer:
                        for tag, model in models.items():
                            dataset = datasets[tag]
                            model.update(dataset)
                            model.optimize(dataset)

                summary_writer = get_tensorboard_writer()
                if summary_writer:
                    with summary_writer.as_default():
                        for tag in datasets:
                            with tf.name_scope(f"{tag}.model"):
                                models[tag].log()
                            tf.summary.scalar(
                                f"{tag}.observation.best_overall",
                                np.min(datasets[tag].observations),
                                step=step,
                            )
                            tf.summary.scalar(
                                f"{tag}.observation.best_new",
                                np.min(tagged_output[tag].observations),
                                step=step,
                            )
                        tf.summary.scalar(
                            "wallclock.step",
                            total_step_wallclock_timer.time,
                            step=step,
                        )
                        tf.summary.scalar(
                            "wallclock.query_point_generation",
                            query_point_generation_timer.time,
                            step=step,
                        )
                        tf.summary.scalar(
                            "wallclock.model_fitting",
                            model_fitting_timer.time
                            + (
                                initial_model_fitting_timer.time
                                if (step == 0 and fit_initial_model)
                                else 0
                            ),
                            step=step,
                        )

            except Exception as error:  # pylint: disable=broad-except
                tf.print(
                    f"\nOptimization failed at step {step}, encountered error with traceback:"
                    f"\n{traceback.format_exc()}"
                    f"\nTerminating optimization and returning the optimization history. You may "
                    f"be able to use the history to restart the process from a previous successful "
                    f"optimization step.\n",
                    output_stream=logging.ERROR,
                )
                if isinstance(error, MemoryError):
                    tf.print(
                        "\nOne possible cause of memory errors is trying to evaluate acquisition "
                        "\nfunctions over large datasets, e.g. when initializing optimizers. "
                        "\nYou may be able to word around this by splitting up the evaluation "
                        "\nusing split_acquisition_function or split_acquisition_function_calls.",
                        output_stream=logging.ERROR,
                    )
                return OptimizationResult(Err(error), history)

        tf.print("Optimization completed without errors", output_stream=logging.INFO)

        record = Record(datasets, models, acquisition_state)
        return OptimizationResult(Ok(record), history)
