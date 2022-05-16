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
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, Generic, MutableMapping, Optional, TypeVar, cast, overload

import absl
import dill
import numpy as np
import tensorflow as tf

from .acquisition.multi_objective import non_dominated

try:
    import pandas as pd
    import seaborn as sns
except ModuleNotFoundError:
    pd = None
    sns = None

from . import logging
from .acquisition.rule import AcquisitionRule, EfficientGlobalOptimization
from .data import Dataset
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
            raise ValueError(f"Expected a single model, found {len(self.models)}")

    def save(self, path: Path | str) -> FrozenRecord[StateType]:
        """Save the record to disk. Will overwrite any existing file at the same path."""
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        with open(path, "wb") as f:
            dill.dump(self, f, dill.HIGHEST_PROTOCOL)
        return FrozenRecord(Path(path))


@dataclass(frozen=True)
class FrozenRecord(Generic[StateType]):
    """
    A Record container saved on disk.

    Note that records are saved via pickling and are therefore neither portable nor secure.
    Only open frozen records generated on the same system.
    """

    path: Path
    """ The path to the pickled Record. """

    def load(self) -> Record[StateType]:
        """Load the record into memory."""
        with open(self.path, "rb") as f:
            return dill.load(f)

    @property
    def datasets(self) -> Mapping[str, Dataset]:
        """The known data from the observer."""
        return self.load().datasets

    @property
    def models(self) -> Mapping[str, TrainableProbabilisticModel]:
        """The models over the :attr:`datasets`."""
        return self.load().models

    @property
    def acquisition_state(self) -> StateType | None:
        """The acquisition state."""
        return self.load().acquisition_state

    @property
    def dataset(self) -> Dataset:
        """The dataset when there is just one dataset."""
        return self.load().dataset

    @property
    def model(self) -> TrainableProbabilisticModel:
        """The model when there is just one dataset."""
        return self.load().model


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

    history: list[Record[StateType] | FrozenRecord[StateType]]
    r"""
    The history of the :class:`Record`\ s from each step of the optimization process. These
    :class:`Record`\ s are created at the *start* of each loop, and as such will never
    include the :attr:`final_result`. The records may be either in memory or on disk.
    """

    @staticmethod
    def step_filename(step: int, num_steps: int) -> str:
        """Default filename for saved optimization steps."""
        return f"step.{step:0{len(str(num_steps - 1))}d}.pickle"

    STEP_GLOB: ClassVar[str] = "step.*.pickle"
    RESULTS_FILENAME: ClassVar[str] = "results.pickle"

    def astuple(
        self,
    ) -> tuple[Result[Record[StateType]], list[Record[StateType] | FrozenRecord[StateType]]]:
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

    def try_get_optimal_point(self) -> tuple[TensorType, TensorType, TensorType]:
        """
        Convenience method to attempt to get the optimal point for a single dataset,
        single objective run.

        :return: Tuple of the optimal query point, observation and its index.
        """
        dataset = self.try_get_final_dataset()
        if tf.rank(dataset.observations) != 2 or dataset.observations.shape[1] != 1:
            raise ValueError("Expected a single objective")
        arg_min_idx = tf.squeeze(tf.argmin(dataset.observations, axis=0))
        return dataset.query_points[arg_min_idx], dataset.observations[arg_min_idx], arg_min_idx

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

    @property
    def loaded_history(self) -> list[Record[StateType]]:
        """The history of the optimization process loaded into memory."""
        return [record if isinstance(record, Record) else record.load() for record in self.history]

    def save_result(self, path: Path | str) -> None:
        """Save the final result to disk. Will overwrite any existing file at the same path."""
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        with open(path, "wb") as f:
            dill.dump(self.final_result, f, dill.HIGHEST_PROTOCOL)

    def save(self, base_path: Path | str) -> None:
        """Save the optimization result to disk. Will overwrite existing files at the same path."""
        path = Path(base_path)
        num_steps = len(self.history)
        self.save_result(path / self.RESULTS_FILENAME)
        for i, record in enumerate(self.loaded_history):
            record_path = path / self.step_filename(i, num_steps)
            record.save(record_path)

    @classmethod
    def from_path(cls, base_path: Path | str) -> OptimizationResult[StateType]:
        """Load a previously saved OptimizationResult."""
        try:
            with open(Path(base_path) / cls.RESULTS_FILENAME, "rb") as f:
                result = dill.load(f)
        except FileNotFoundError as e:
            result = Err(e)

        history: list[Record[StateType] | FrozenRecord[StateType]] = [
            FrozenRecord(file) for file in sorted(Path(base_path).glob(cls.STEP_GLOB))
        ]
        return cls(result, history)


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
        track_path: Optional[Path | str] = None,
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
        track_path: Optional[Path | str] = None,
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
        track_path: Optional[Path | str] = None,
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
        track_path: Optional[Path | str] = None,
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
        track_path: Optional[Path | str] = None,
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
        track_path: Optional[Path | str] = None,
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
        track_path: Optional[Path | str] = None,
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
        track_path: Optional[Path | str] = None,
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
        track_path: Optional[Path | str] = None,
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
        track_path: Optional[Path | str] = None,
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
        track_path: Optional[Path | str] = None,
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
        them instead and print a message (using `absl` at level `absl.logging.ERROR`).
        If ``track_state`` is enabled, then in addition to the final result, the history of the
        optimization process will also be returned. If ``track_path`` is also set, then
        the history and final result will be saved to disk rather than all being kept in memory.

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
        :param track_path: If set, the optimization state is saved to disk at this path,
            rather than being copied in memory.
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

        history: list[FrozenRecord[StateType] | Record[StateType]] = []
        query_plot_dfs: dict[int, pd.DataFrame] = {}
        observation_plot_dfs: dict[str, pd.DataFrame] = {}

        summary_writer = logging.get_tensorboard_writer()
        if summary_writer:
            with summary_writer.as_default(step=0):
                self._write_summary_init(
                    acquisition_rule, datasets, models, num_steps, observation_plot_dfs
                )

        for step in range(num_steps):
            logging.set_step_number(step)
            try:

                if track_state:
                    try:
                        if track_path is None:
                            datasets_copy = copy.deepcopy(datasets)
                            models_copy = copy.deepcopy(models)
                            acquisition_state_copy = copy.deepcopy(acquisition_state)
                            record = Record(datasets_copy, models_copy, acquisition_state_copy)
                            history.append(record)
                        else:
                            track_path = Path(track_path)
                            record = Record(datasets, models, acquisition_state)
                            file_name = OptimizationResult.step_filename(step, num_steps)
                            history.append(record.save(track_path / file_name))
                    except Exception as e:
                        raise NotImplementedError(
                            "Failed to save the optimization state. Some models do not support "
                            "deecopying or serialization and cannot be saved. "
                            "(This is particularly common for deep neural network models.) "
                            "For these models, the `track_state`` argument of the "
                            ":meth:`~trieste.bayesian_optimizer.BayesianOptimizer.optimize` method "
                            "should be set to `False`. This means that only the final model "
                            "will be available."
                        ) from e

                with Timer() as total_step_wallclock_timer:
                    with Timer() as initial_model_fitting_timer:
                        if step == 0 and fit_initial_model:
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

                if summary_writer:
                    with summary_writer.as_default(step=step):
                        self._write_summary_step(
                            datasets,
                            models,
                            query_points,
                            tagged_output,
                            initial_model_fitting_timer,
                            model_fitting_timer,
                            query_point_generation_timer,
                            total_step_wallclock_timer,
                            observation_plot_dfs,
                            query_plot_dfs,
                        )

            except Exception as error:  # pylint: disable=broad-except
                tf.print(
                    f"\nOptimization failed at step {step}, encountered error with traceback:"
                    f"\n{traceback.format_exc()}"
                    f"\nTerminating optimization and returning the optimization history. You may "
                    f"be able to use the history to restart the process from a previous successful "
                    f"optimization step.\n",
                    output_stream=absl.logging.ERROR,
                )
                if isinstance(error, MemoryError):
                    tf.print(
                        "\nOne possible cause of memory errors is trying to evaluate acquisition "
                        "\nfunctions over large datasets, e.g. when initializing optimizers. "
                        "\nYou may be able to word around this by splitting up the evaluation "
                        "\nusing split_acquisition_function or split_acquisition_function_calls.",
                        output_stream=absl.logging.ERROR,
                    )
                result = OptimizationResult(Err(error), history)
                if track_state and track_path is not None:
                    result.save_result(Path(track_path) / OptimizationResult.RESULTS_FILENAME)
                return result

        tf.print("Optimization completed without errors", output_stream=absl.logging.INFO)

        record = Record(datasets, models, acquisition_state)
        result = OptimizationResult(Ok(record), history)
        if track_state and track_path is not None:
            result.save_result(Path(track_path) / OptimizationResult.RESULTS_FILENAME)
        return result

    def _write_summary_init(
        self,
        acquisition_rule: AcquisitionRule[
            TensorType | State[StateType | None, TensorType],
            SearchSpaceType,
            TrainableProbabilisticModelType,
        ],
        datasets: Mapping[str, Dataset],
        models: Mapping[str, TrainableProbabilisticModel],
        num_steps: int,
        observation_plot_dfs: MutableMapping[str, pd.DataFrame],
    ) -> None:
        """Write initial TensorBoard summary (and set up any initial monitoring state)."""
        devices = tf.config.list_logical_devices()
        logging.text(
            "metadata",
            f"Observer: `{self._observer}`\n\n"
            f"Number of steps: `{num_steps}`\n\n"
            f"Number of initial points: "
            f"`{dict((k, len(v)) for k, v in datasets.items())}`\n\n"
            f"Search Space: `{self._search_space}`\n\n"
            f"Acquisition rule:\n\n    {acquisition_rule}\n\n"
            f"Models:\n\n    {models}\n\n"
            f"Available devices: `{dict(Counter(d.device_type for d in devices))}`",
        )

        seaborn_warning = False
        if logging.include_summary("query_points/_pairplot") and not (pd and sns):
            seaborn_warning = True
        for tag in datasets:
            if logging.include_summary(f"{tag}.observations/_pairplot"):
                output_dim = tf.shape(datasets[tag].observations)[-1]
                if output_dim >= 2:
                    if not (pd and sns):
                        seaborn_warning = True
                    else:
                        columns = [f"x{i}" for i in range(output_dim)]
                        observation_plot_dfs[tag] = pd.DataFrame(
                            datasets[tag].observations, columns=columns
                        ).applymap(float)
                        observation_plot_dfs[tag]["observations"] = "initial"

        if seaborn_warning:
            tf.print(
                "\nPairplot TensorBoard summaries require seaborn to be installed."
                "\nOne way to do this is to install 'trieste[plotting]'.",
                output_stream=absl.logging.INFO,
            )

    def _write_summary_step(
        self,
        datasets: Mapping[str, Dataset],
        models: Mapping[str, TrainableProbabilisticModel],
        query_points: TensorType,
        tagged_output: Mapping[str, TensorType],
        initial_model_fitting_timer: Timer,
        model_fitting_timer: Timer,
        query_point_generation_timer: Timer,
        total_step_wallclock_timer: Timer,
        observation_plot_dfs: MutableMapping[str, pd.DataFrame],
        query_plot_dfs: MutableMapping[int, pd.DataFrame],
    ) -> None:
        """Write TensorBoard summary for the current step."""
        for tag in datasets:

            with tf.name_scope(f"{tag}.model"):
                models[tag].log()

            output_dim = tf.shape(tagged_output[tag].observations)[-1]
            for i in tf.range(output_dim):
                suffix = f"[{i}]" if output_dim > 1 else ""
                if tf.size(tagged_output[tag].observations) > 0:
                    logging.histogram(
                        f"{tag}.observation{suffix}/new_observations",
                        tagged_output[tag].observations[..., i],
                    )
                    logging.scalar(
                        f"{tag}.observation{suffix}/best_new_observation",
                        np.min(tagged_output[tag].observations[..., i]),
                    )
                if tf.size(datasets[tag].observations) > 0:
                    logging.scalar(
                        f"{tag}.observation{suffix}/best_overall",
                        np.min(datasets[tag].observations[..., i]),
                    )

            if logging.include_summary(f"{tag}.observations/_pairplot") and (
                pd and sns and output_dim >= 2
            ):
                columns = [f"x{i}" for i in range(output_dim)]
                observation_new_df = pd.DataFrame(
                    tagged_output[tag].observations, columns=columns
                ).applymap(float)
                observation_new_df["observations"] = "new"
                observation_plot_df = pd.concat(
                    (observation_plot_dfs.get(tag), observation_new_df),
                    copy=False,
                    ignore_index=True,
                )
                observation_plot_df["pareto"] = non_dominated(datasets[tag].observations)[1] == 0
                observation_plot_df["observation type"] = observation_plot_df.apply(
                    lambda x: x["observations"] + x["pareto"] * " (non-dominated)",
                    axis=1,
                )
                pairplot = sns.pairplot(
                    observation_plot_df,
                    vars=columns,
                    hue="observation type",
                    hue_order=[
                        "initial",
                        "old",
                        "new",
                        "initial (non-dominated)",
                        "old (non-dominated)",
                        "new (non-dominated)",
                    ],
                    palette={
                        "initial": "tab:green",
                        "old": "tab:green",
                        "new": "tab:orange",
                        "initial (non-dominated)": "tab:purple",
                        "old (non-dominated)": "tab:purple",
                        "new (non-dominated)": "tab:red",
                    },
                    markers={
                        "initial": "X",
                        "old": "o",
                        "new": "o",
                        "initial (non-dominated)": "X",
                        "old (non-dominated)": "o",
                        "new (non-dominated)": "o",
                    },
                )
                logging.pyplot(f"{tag}.observations/_pairplot", pairplot.fig)
                observation_plot_df.loc[
                    observation_plot_df["observations"] == "new", "observations"
                ] = "old"
                observation_plot_dfs[tag] = observation_plot_df

        if tf.rank(query_points) == 2:
            for i in tf.range(tf.shape(query_points)[1]):
                if len(query_points) == 1:
                    logging.scalar(f"query_points/[{i}]", float(query_points[0, i]))
                else:
                    logging.histogram(f"query_points/[{i}]", query_points[:, i])

        if pd and sns and logging.include_summary("query_points/_pairplot"):
            columns = [f"x{i}" for i in range(tf.shape(query_points)[1])]
            query_new_df = pd.DataFrame(query_points, columns=columns).applymap(float)
            query_new_df["query points"] = "new"
            query_plot_df = pd.concat(
                (query_plot_dfs.get(0), query_new_df), copy=False, ignore_index=True
            )
            pairplot = sns.pairplot(query_plot_df, hue="query points")
            padding = 0.025 * (self._search_space.upper - self._search_space.lower)
            upper_limits = self._search_space.upper + padding
            lower_limits = self._search_space.lower - padding
            for i in range(self._search_space.dimension):
                pairplot.axes[0, i].set_xlim((lower_limits[i], upper_limits[i]))
                pairplot.axes[i, 0].set_ylim((lower_limits[i], upper_limits[i]))
            logging.pyplot("query_points/_pairplot", pairplot.fig)
            query_plot_df["query points"] = "old"
            query_plot_dfs[0] = query_plot_df

        logging.scalar("wallclock/step", total_step_wallclock_timer.time)
        logging.scalar(
            "wallclock/query_point_generation",
            query_point_generation_timer.time,
        )
        logging.scalar(
            "wallclock/model_fitting",
            model_fitting_timer.time + initial_model_fitting_timer.time,
        )
