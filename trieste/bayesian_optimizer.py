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
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    Mapping,
    MutableMapping,
    Optional,
    TypeVar,
    cast,
    overload,
)

import absl
import dill
import numpy as np
import tensorflow as tf
from scipy.spatial.distance import pdist

from .acquisition.multi_objective import non_dominated
from .models.utils import optimize_model_and_save_result

try:
    import pandas as pd
    import seaborn as sns
except ModuleNotFoundError:
    pd = None
    sns = None

from . import logging
from .acquisition.rule import (
    AcquisitionRule,
    EfficientGlobalOptimization,
    LocalDatasetsAcquisitionRule,
)
from .acquisition.utils import with_local_datasets
from .data import Dataset
from .models import (
    ProbabilisticModel,
    SupportsCovarianceWithTopFidelity,
    TrainableProbabilisticModel,
)
from .objectives.utils import mk_batch_observer
from .observer import OBJECTIVE, Observer
from .space import SearchSpace
from .types import State, Tag, TensorType
from .utils import Err, Ok, Result, Timer
from .utils.misc import LocalizedTag, get_value_for_tag, ignoring_local_tags

StateType = TypeVar("StateType")
""" Unbound type variable. """

SearchSpaceType = TypeVar("SearchSpaceType", bound=SearchSpace)
""" Type variable bound to :class:`SearchSpace`. """

ProbabilisticModelType = TypeVar(
    "ProbabilisticModelType",
    bound=ProbabilisticModel,
    covariant=True,
)
""" Covariant type variable bound to :class:`ProbabilisticModel`. """

TrainableProbabilisticModelType = TypeVar(
    "TrainableProbabilisticModelType", bound=TrainableProbabilisticModel, contravariant=True
)
""" Contravariant type variable bound to :class:`TrainableProbabilisticModel`. """

EarlyStopCallback = Callable[
    [Mapping[Tag, Dataset], Mapping[Tag, TrainableProbabilisticModelType], Optional[StateType]],
    bool,
]
""" Early stop callback type, generic in the model and state types. """


@dataclass(frozen=True)
class Record(Generic[StateType, ProbabilisticModelType]):
    """Container to record the state of each step of the optimization process."""

    datasets: Mapping[Tag, Dataset]
    """ The known data from the observer. """

    models: Mapping[Tag, ProbabilisticModelType]
    """ The models over the :attr:`datasets`. """

    acquisition_state: StateType | None
    """ The acquisition state. """

    @property
    def dataset(self) -> Dataset:
        """The dataset when there is just one dataset."""
        # Ignore local datasets.
        datasets: Mapping[Tag, Dataset] = ignoring_local_tags(self.datasets)
        if len(datasets) == 1:
            return next(iter(datasets.values()))
        else:
            raise ValueError(f"Expected a single dataset, found {len(datasets)}")

    @property
    def model(self) -> ProbabilisticModelType:
        """The model when there is just one dataset."""
        # Ignore local models.
        models: Mapping[Tag, ProbabilisticModelType] = ignoring_local_tags(self.models)
        if len(models) == 1:
            return next(iter(models.values()))
        else:
            raise ValueError(f"Expected a single model, found {len(models)}")

    def save(self, path: Path | str) -> FrozenRecord[StateType, ProbabilisticModelType]:
        """Save the record to disk. Will overwrite any existing file at the same path."""
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        with open(path, "wb") as f:
            dill.dump(self, f, dill.HIGHEST_PROTOCOL)
        return FrozenRecord(Path(path))


@dataclass(frozen=True)
class FrozenRecord(Generic[StateType, ProbabilisticModelType]):
    """
    A Record container saved on disk.

    Note that records are saved via pickling and are therefore neither portable nor secure.
    Only open frozen records generated on the same system.
    """

    path: Path
    """ The path to the pickled Record. """

    def load(self) -> Record[StateType, ProbabilisticModelType]:
        """Load the record into memory."""
        with open(self.path, "rb") as f:
            return dill.load(f)

    @property
    def datasets(self) -> Mapping[Tag, Dataset]:
        """The known data from the observer."""
        return self.load().datasets

    @property
    def models(self) -> Mapping[Tag, ProbabilisticModelType]:
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
    def model(self) -> ProbabilisticModelType:
        """The model when there is just one dataset."""
        return self.load().model


# this should be a generic NamedTuple, but mypy doesn't support them
#  https://github.com/python/mypy/issues/685
@dataclass(frozen=True)
class OptimizationResult(Generic[StateType, ProbabilisticModelType]):
    """The final result, and the historical data of the optimization process."""

    final_result: Result[Record[StateType, ProbabilisticModelType]]
    """
    The final result of the optimization process. This contains either a :class:`Record` or an
    exception.
    """

    history: list[
        Record[StateType, ProbabilisticModelType] | FrozenRecord[StateType, ProbabilisticModelType]
    ]
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
    ) -> tuple[
        Result[Record[StateType, ProbabilisticModelType]],
        list[
            Record[StateType, ProbabilisticModelType]
            | FrozenRecord[StateType, ProbabilisticModelType]
        ],
    ]:
        """
        **Note:** In contrast to the standard library function :func:`dataclasses.astuple`, this
        method does *not* deepcopy instance attributes.

        :return: The :attr:`final_result` and :attr:`history` as a 2-tuple.
        """
        return self.final_result, self.history

    @property
    def is_ok(self) -> bool:
        """`True` if the final result contains a :class:`Record`."""
        return self.final_result.is_ok

    @property
    def is_err(self) -> bool:
        """`True` if the final result contains an exception."""
        return self.final_result.is_err

    def try_get_final_datasets(self) -> Mapping[Tag, Dataset]:
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
        # Ignore local datasets.
        datasets = ignoring_local_tags(datasets)
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
        if tf.reduce_any(
            [
                isinstance(model, SupportsCovarianceWithTopFidelity)
                for model in self.try_get_final_models()
            ]
        ):
            raise ValueError("Expected single fidelity models")
        arg_min_idx = tf.squeeze(tf.argmin(dataset.observations, axis=0))
        return dataset.query_points[arg_min_idx], dataset.observations[arg_min_idx], arg_min_idx

    def try_get_final_models(self) -> Mapping[Tag, ProbabilisticModelType]:
        """
        Convenience method to attempt to get the final models.

        :return: The final models, if the optimization completed successfully.
        :raise Exception: If an exception occurred during optimization.
        """
        return self.final_result.unwrap().models

    def try_get_final_model(self) -> ProbabilisticModelType:
        """
        Convenience method to attempt to get the final model for a single model run.

        :return: The final model, if the optimization completed successfully.
        :raise Exception: If an exception occurred during optimization.
        :raise ValueError: If the optimization was not a single model run.
        """
        models = self.try_get_final_models()
        # Ignore local models.
        models = ignoring_local_tags(models)
        if len(models) == 1:
            return next(iter(models.values()))
        else:
            raise ValueError(f"Expected single model, found {len(models)}")

    @property
    def loaded_history(self) -> list[Record[StateType, ProbabilisticModelType]]:
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
    def from_path(
        cls, base_path: Path | str
    ) -> OptimizationResult[StateType, ProbabilisticModelType]:
        """Load a previously saved OptimizationResult."""
        try:
            with open(Path(base_path) / cls.RESULTS_FILENAME, "rb") as f:
                result = dill.load(f)
        except FileNotFoundError as e:
            result = Err(e)

        history: list[
            Record[StateType, ProbabilisticModelType]
            | FrozenRecord[StateType, ProbabilisticModelType]
        ] = [FrozenRecord(file) for file in sorted(Path(base_path).glob(cls.STEP_GLOB))]
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
        datasets: Mapping[Tag, Dataset],
        models: Mapping[Tag, TrainableProbabilisticModel],
        *,
        track_state: bool = True,
        track_path: Optional[Path | str] = None,
        fit_model: bool = True,
        fit_initial_model: bool = True,
        early_stop_callback: Optional[
            EarlyStopCallback[TrainableProbabilisticModel, object]
        ] = None,
        start_step: int = 0,
    ) -> OptimizationResult[None, TrainableProbabilisticModel]: ...

    @overload
    def optimize(
        self,
        num_steps: int,
        datasets: Mapping[Tag, Dataset],
        models: Mapping[Tag, TrainableProbabilisticModelType],
        acquisition_rule: AcquisitionRule[
            TensorType, SearchSpaceType, TrainableProbabilisticModelType
        ],
        *,
        track_state: bool = True,
        track_path: Optional[Path | str] = None,
        fit_model: bool = True,
        fit_initial_model: bool = True,
        early_stop_callback: Optional[
            EarlyStopCallback[TrainableProbabilisticModelType, object]
        ] = None,
        start_step: int = 0,
        # this should really be OptimizationResult[None], but tf.Tensor is untyped so the type
        # checker can't differentiate between TensorType and State[S | None, TensorType], and
        # the return types clash. object is close enough to None that object will do.
    ) -> OptimizationResult[object, TrainableProbabilisticModelType]: ...

    @overload
    def optimize(
        self,
        num_steps: int,
        datasets: Mapping[Tag, Dataset],
        models: Mapping[Tag, TrainableProbabilisticModelType],
        acquisition_rule: AcquisitionRule[
            TensorType, SearchSpaceType, TrainableProbabilisticModelType
        ],
        *,
        track_state: bool = True,
        track_path: Optional[Path | str] = None,
        fit_model: bool = True,
        fit_initial_model: bool = True,
        early_stop_callback: Optional[
            EarlyStopCallback[TrainableProbabilisticModelType, object]
        ] = None,
        start_step: int = 0,
    ) -> OptimizationResult[object, TrainableProbabilisticModelType]: ...

    @overload
    def optimize(
        self,
        num_steps: int,
        datasets: Mapping[Tag, Dataset],
        models: Mapping[Tag, TrainableProbabilisticModelType],
        acquisition_rule: AcquisitionRule[
            State[StateType | None, TensorType], SearchSpaceType, TrainableProbabilisticModelType
        ],
        acquisition_state: StateType | None = None,
        *,
        track_state: bool = True,
        track_path: Optional[Path | str] = None,
        fit_model: bool = True,
        fit_initial_model: bool = True,
        early_stop_callback: Optional[
            EarlyStopCallback[TrainableProbabilisticModelType, StateType]
        ] = None,
        start_step: int = 0,
    ) -> OptimizationResult[StateType, TrainableProbabilisticModelType]: ...

    @overload
    def optimize(
        self,
        num_steps: int,
        datasets: Mapping[Tag, Dataset],
        models: Mapping[Tag, TrainableProbabilisticModelType],
        acquisition_rule: AcquisitionRule[
            State[StateType | None, TensorType], SearchSpaceType, TrainableProbabilisticModelType
        ],
        acquisition_state: StateType | None = None,
        *,
        track_state: bool = True,
        track_path: Optional[Path | str] = None,
        fit_model: bool = True,
        fit_initial_model: bool = True,
        early_stop_callback: Optional[
            EarlyStopCallback[TrainableProbabilisticModelType, StateType]
        ] = None,
        start_step: int = 0,
    ) -> OptimizationResult[StateType, TrainableProbabilisticModelType]: ...

    @overload
    def optimize(
        self,
        num_steps: int,
        datasets: Dataset,
        models: TrainableProbabilisticModel,
        *,
        track_state: bool = True,
        track_path: Optional[Path | str] = None,
        fit_model: bool = True,
        fit_initial_model: bool = True,
        early_stop_callback: Optional[
            EarlyStopCallback[TrainableProbabilisticModel, object]
        ] = None,
        start_step: int = 0,
    ) -> OptimizationResult[None, TrainableProbabilisticModel]: ...

    @overload
    def optimize(
        self,
        num_steps: int,
        datasets: Dataset,
        models: TrainableProbabilisticModelType,
        acquisition_rule: AcquisitionRule[
            TensorType, SearchSpaceType, TrainableProbabilisticModelType
        ],
        *,
        track_state: bool = True,
        track_path: Optional[Path | str] = None,
        fit_model: bool = True,
        fit_initial_model: bool = True,
        early_stop_callback: Optional[
            EarlyStopCallback[TrainableProbabilisticModelType, object]
        ] = None,
        start_step: int = 0,
    ) -> OptimizationResult[object, TrainableProbabilisticModelType]: ...

    @overload
    def optimize(
        self,
        num_steps: int,
        datasets: Dataset,
        models: TrainableProbabilisticModelType,
        acquisition_rule: AcquisitionRule[
            TensorType, SearchSpaceType, TrainableProbabilisticModelType
        ],
        *,
        track_state: bool = True,
        track_path: Optional[Path | str] = None,
        fit_model: bool = True,
        fit_initial_model: bool = True,
        early_stop_callback: Optional[
            EarlyStopCallback[TrainableProbabilisticModelType, object]
        ] = None,
        start_step: int = 0,
    ) -> OptimizationResult[object, TrainableProbabilisticModelType]: ...

    @overload
    def optimize(
        self,
        num_steps: int,
        datasets: Dataset,
        models: TrainableProbabilisticModelType,
        acquisition_rule: AcquisitionRule[
            State[StateType | None, TensorType], SearchSpaceType, TrainableProbabilisticModelType
        ],
        acquisition_state: StateType | None = None,
        *,
        track_state: bool = True,
        track_path: Optional[Path | str] = None,
        fit_model: bool = True,
        fit_initial_model: bool = True,
        early_stop_callback: Optional[
            EarlyStopCallback[TrainableProbabilisticModelType, StateType]
        ] = None,
        start_step: int = 0,
    ) -> OptimizationResult[StateType, TrainableProbabilisticModelType]: ...

    @overload
    def optimize(
        self,
        num_steps: int,
        datasets: Dataset,
        models: TrainableProbabilisticModelType,
        acquisition_rule: AcquisitionRule[
            State[StateType | None, TensorType], SearchSpaceType, TrainableProbabilisticModelType
        ],
        acquisition_state: StateType | None = None,
        *,
        track_state: bool = True,
        track_path: Optional[Path | str] = None,
        fit_model: bool = True,
        fit_initial_model: bool = True,
        early_stop_callback: Optional[
            EarlyStopCallback[TrainableProbabilisticModelType, StateType]
        ] = None,
        start_step: int = 0,
    ) -> OptimizationResult[StateType, TrainableProbabilisticModelType]: ...

    def optimize(
        self,
        num_steps: int,
        datasets: Mapping[Tag, Dataset] | Dataset,
        models: Mapping[Tag, TrainableProbabilisticModelType] | TrainableProbabilisticModelType,
        acquisition_rule: (
            AcquisitionRule[
                TensorType | State[StateType | None, TensorType],
                SearchSpaceType,
                TrainableProbabilisticModelType,
            ]
            | None
        ) = None,
        acquisition_state: StateType | None = None,
        *,
        track_state: bool = True,
        track_path: Optional[Path | str] = None,
        fit_model: bool = True,
        fit_initial_model: bool = True,
        early_stop_callback: Optional[
            EarlyStopCallback[TrainableProbabilisticModelType, StateType]
        ] = None,
        start_step: int = 0,
    ) -> (
        OptimizationResult[StateType, TrainableProbabilisticModelType]
        | OptimizationResult[None, TrainableProbabilisticModelType]
    ):
        """
        Attempt to find the minimizer of the ``observer`` in the ``search_space`` (both specified at
        :meth:`__init__`). This is the central implementation of the Bayesian optimization loop.

        For each step in ``num_steps``, this method:
            - Finds the next points with which to query the ``observer`` using the
              ``acquisition_rule``'s :meth:`acquire` method, passing it the ``search_space``,
              ``datasets``, ``models``, and current acquisition state.
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
        :param models: The model to use for each :class:`~trieste.data.Dataset` in
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
        :param fit_model: If `False` then we never fit the model during BO (e.g. if we
            are using a rule that doesn't rely on the models and don't want to waste computation).
        :param fit_initial_model: If `False` then we assume that the initial models have
            already been optimized on the datasets and so do not require optimization before
            the first optimization step.
        :param early_stop_callback: An optional callback that is evaluated with the current
            datasets, models and optimization state before every optimization step. If this
            returns `True` then the optimization loop is terminated early.
        :param start_step: The step number to start with. This number is removed from ``num_steps``
            and is useful for restarting previous computations.
        :return: An :class:`OptimizationResult`. The :attr:`final_result` element contains either
            the final optimization data, models and acquisition state, or, if an exception was
            raised while executing the optimization loop, it contains the exception raised. In
            either case, the :attr:`history` element is the history of the data, models and
            acquisition state at the *start* of each optimization step (up to and including any step
            that fails to complete). The history will never include the final optimization result.
        :raise ValueError: If any of the following are true:

            - ``num_steps`` is negative.
            - the keys in ``datasets`` and ``models`` do not match
            - ``datasets`` or ``models`` are empty
            - the default `acquisition_rule` is used and the tags are not `OBJECTIVE`.
        """
        # Copy the dataset so we don't change the one provided by the user.
        datasets = copy.deepcopy(datasets)

        if isinstance(datasets, Dataset):
            datasets = {OBJECTIVE: datasets}
        if not isinstance(models, Mapping):
            models = {OBJECTIVE: models}

        filtered_datasets = datasets
        # reassure the type checker that everything is tagged
        datasets = cast(Dict[Tag, Dataset], datasets)
        models = cast(Dict[Tag, TrainableProbabilisticModelType], models)

        if num_steps < 0:
            raise ValueError(f"num_steps must be at least 0, got {num_steps}")

        # Get set of dataset and model keys, ignoring any local tag index. That is, only the
        # global tag part is considered.
        datasets_keys = {LocalizedTag.from_tag(tag).global_tag for tag in datasets.keys()}
        models_keys = {LocalizedTag.from_tag(tag).global_tag for tag in models.keys()}
        if datasets_keys != models_keys:
            raise ValueError(
                f"datasets and models should contain the same keys. Got {datasets_keys} and"
                f" {models_keys} respectively."
            )

        if not datasets:
            raise ValueError("dicts of datasets and models must be populated.")

        if acquisition_rule is None:
            if datasets.keys() != {OBJECTIVE}:
                raise ValueError(
                    f"Default acquisition rule EfficientGlobalOptimization requires tag"
                    f" {OBJECTIVE!r}, got keys {datasets.keys()}"
                )

            acquisition_rule = EfficientGlobalOptimization[
                SearchSpaceType, TrainableProbabilisticModelType
            ]()

        history: list[
            FrozenRecord[StateType, TrainableProbabilisticModelType]
            | Record[StateType, TrainableProbabilisticModelType]
        ] = []
        history_state = acquisition_state
        query_plot_dfs: dict[int, pd.DataFrame] = {}
        observation_plot_dfs = observation_plot_init(datasets)

        summary_writer = logging.get_tensorboard_writer()
        if summary_writer:
            with summary_writer.as_default(step=0):
                write_summary_init(
                    self._observer,
                    self._search_space,
                    acquisition_rule,
                    datasets,
                    models,
                    num_steps,
                )

        for step in range(start_step + 1, num_steps + 1):
            logging.set_step_number(step)

            if early_stop_callback and early_stop_callback(datasets, models, acquisition_state):
                tf.print("Optimization terminated early", output_stream=absl.logging.INFO)
                break

            try:
                if track_state:
                    try:
                        # note that we use the state as it was at acquisition time, not the one
                        # modified in filter_datasets in preparation for the next acquisition,
                        # so we can restart the optimization correctly (and also for plotting)
                        if track_path is None:
                            datasets_copy = copy.deepcopy(datasets)
                            models_copy = copy.deepcopy(models)
                            history_state_copy = copy.deepcopy(history_state)
                            record = Record(datasets_copy, models_copy, history_state_copy)
                            history.append(record)
                        else:
                            track_path = Path(track_path)
                            record = Record(datasets, models, history_state)
                            file_name = OptimizationResult.step_filename(step, num_steps)
                            history.append(record.save(track_path / file_name))
                    except Exception as e:
                        raise NotImplementedError(
                            "Failed to save the optimization state. Some models do not support "
                            "deecopying or serialization and cannot be saved. "
                            "(This is particularly common for deep neural network models, though "
                            "some of the model wrappers accept a model closure as a workaround.) "
                            "For these models, the `track_state`` argument of the "
                            ":meth:`~trieste.bayesian_optimizer.BayesianOptimizer.optimize` method "
                            "should be set to `False`. This means that only the final model "
                            "will be available."
                        ) from e

                if step == 1:
                    # See explanation in AskTellOptimizer.__init__().
                    if isinstance(acquisition_rule, LocalDatasetsAcquisitionRule):
                        datasets = with_local_datasets(
                            datasets, acquisition_rule.num_local_datasets
                        )
                        acquisition_rule.initialize_subspaces(self._search_space)

                    filtered_datasets_or_callable: (
                        Mapping[Tag, Dataset] | State[StateType | None, Mapping[Tag, Dataset]]
                    ) = acquisition_rule.filter_datasets(models, datasets)
                    if callable(filtered_datasets_or_callable):
                        acquisition_state, filtered_datasets = filtered_datasets_or_callable(
                            acquisition_state
                        )
                    else:
                        filtered_datasets = filtered_datasets_or_callable

                    if fit_model and fit_initial_model:
                        with Timer() as initial_model_fitting_timer:
                            for tag, model in models.items():
                                # Prefer local dataset if available.
                                tags = [tag, LocalizedTag.from_tag(tag).global_tag]
                                _, dataset = get_value_for_tag(filtered_datasets, *tags)
                                assert dataset is not None
                                model.update(dataset)
                                optimize_model_and_save_result(model, dataset)
                        if summary_writer:
                            logging.set_step_number(0)
                            with summary_writer.as_default(step=0):
                                write_summary_initial_model_fit(
                                    datasets, models, initial_model_fitting_timer
                                )
                            logging.set_step_number(step)

                with Timer() as total_step_wallclock_timer:
                    with Timer() as query_point_generation_timer:
                        points_or_stateful = acquisition_rule.acquire(
                            self._search_space, models, datasets=filtered_datasets
                        )
                        if callable(points_or_stateful):
                            acquisition_state, query_points = points_or_stateful(acquisition_state)
                            history_state = acquisition_state
                        else:
                            query_points = points_or_stateful

                    observer = self._observer
                    # If query_points are rank 3, then use a batched observer.
                    if tf.rank(query_points) == 3:
                        observer = mk_batch_observer(observer)
                    observer_output = observer(query_points)

                    tagged_output = (
                        observer_output
                        if isinstance(observer_output, Mapping)
                        else {OBJECTIVE: observer_output}
                    )

                    for tag, new_dataset in tagged_output.items():
                        datasets[tag] += new_dataset

                    filtered_datasets_or_callable = acquisition_rule.filter_datasets(
                        models, datasets
                    )
                    if callable(filtered_datasets_or_callable):
                        acquisition_state, filtered_datasets = filtered_datasets_or_callable(
                            acquisition_state
                        )
                    else:
                        filtered_datasets = filtered_datasets_or_callable

                    with Timer() as model_fitting_timer:
                        if fit_model:
                            for tag, model in models.items():
                                # Always use the matching dataset to the model. If the model is
                                # local, then the dataset should be too by this stage.
                                dataset = filtered_datasets[tag]
                                model.update(dataset)
                                optimize_model_and_save_result(model, dataset)

                if summary_writer:
                    with summary_writer.as_default(step=step):
                        write_summary_observations(
                            datasets,
                            models,
                            tagged_output,
                            model_fitting_timer,
                            observation_plot_dfs,
                        )
                        write_summary_query_points(
                            datasets,
                            models,
                            self._search_space,
                            query_points,
                            query_point_generation_timer,
                            query_plot_dfs,
                        )
                        logging.scalar("wallclock/step", total_step_wallclock_timer.time)

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

        record = Record(datasets, models, history_state)
        result = OptimizationResult(Ok(record), history)
        if track_state and track_path is not None:
            result.save_result(Path(track_path) / OptimizationResult.RESULTS_FILENAME)
        return result

    def continue_optimization(
        self,
        num_steps: int,
        optimization_result: OptimizationResult[StateType, TrainableProbabilisticModelType],
        *args: Any,
        **kwargs: Any,
    ) -> OptimizationResult[StateType, TrainableProbabilisticModelType]:
        """
        Continue a previous optimization that either failed, was terminated early, or which
        you simply wish to run for more steps.

        :param num_steps: The total number of optimization steps, including any that have already
            been run.
        :param optimization_result: The optimization result from which to extract the datasets,
            models and acquisition state. If the result was successful then the final result is
            used; otherwise the last record in the history is used. The size of the history
            is used to determine how many more steps are required.
        :param args: Any more positional arguments to pass on to optimize.
        :param kwargs: Any more keyword arguments to pass on to optimize.
        :return: An :class:`OptimizationResult`. The history will contain both the history from
            `optimization_result` (including the `final_result` if that was successful) and
            any new records.
        """
        history: list[
            Record[StateType, TrainableProbabilisticModelType]
            | FrozenRecord[StateType, TrainableProbabilisticModelType]
        ] = []
        history.extend(optimization_result.history)
        if optimization_result.final_result.is_ok:
            history.append(optimization_result.final_result.unwrap())
        if not history:
            raise ValueError("Cannot continue from empty optimization result")

        result = self.optimize(  # type: ignore[call-overload]
            num_steps,
            history[-1].datasets,
            history[-1].models,
            *args,
            acquisition_state=history[-1].acquisition_state,
            **kwargs,
            start_step=len(history) - 1,
        )
        result.history[:1] = history
        return result


def write_summary_init(
    observer: Observer,
    search_space: SearchSpace,
    acquisition_rule: AcquisitionRule[
        TensorType | State[StateType | None, TensorType],
        SearchSpaceType,
        TrainableProbabilisticModelType,
    ],
    datasets: Mapping[Tag, Dataset],
    models: Mapping[Tag, TrainableProbabilisticModel],
    num_steps: int,
) -> None:
    """Write initial BO loop TensorBoard summary."""
    devices = tf.config.list_logical_devices()
    logging.text(
        "metadata",
        f"Observer: `{observer}`\n\n"
        f"Number of steps: `{num_steps}`\n\n"
        f"Number of initial points: "
        f"`{dict((k, len(v)) for k, v in datasets.items())}`\n\n"
        f"Search Space: `{search_space}`\n\n"
        f"Acquisition rule:\n\n    {acquisition_rule}\n\n"
        f"Models:\n\n    {models}\n\n"
        f"Available devices: `{dict(Counter(d.device_type for d in devices))}`",
    )


def write_summary_initial_model_fit(
    datasets: Mapping[Tag, Dataset],
    models: Mapping[Tag, ProbabilisticModel],
    model_fitting_timer: Timer,
) -> None:
    """Write TensorBoard summary for the model fitting to the initial data."""
    for tag, model in models.items():
        with tf.name_scope(f"{tag}.model"):
            # Prefer local dataset if available.
            tags = [tag, LocalizedTag.from_tag(tag).global_tag]
            _, dataset = get_value_for_tag(datasets, *tags)
            assert dataset is not None
            model.log(dataset)
    logging.scalar(
        "wallclock/model_fitting",
        model_fitting_timer.time,
    )


def observation_plot_init(
    datasets: Mapping[Tag, Dataset],
) -> dict[Tag, pd.DataFrame]:
    """Initialise query point pairplot dataframes with initial observations.
    Also logs warnings if pairplot dependencies are not installed."""
    observation_plot_dfs: dict[Tag, pd.DataFrame] = {}
    if logging.get_tensorboard_writer():
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
    return observation_plot_dfs


def write_summary_observations(
    datasets: Mapping[Tag, Dataset],
    models: Mapping[Tag, ProbabilisticModel],
    tagged_output: Mapping[Tag, TensorType],
    model_fitting_timer: Timer,
    observation_plot_dfs: MutableMapping[Tag, pd.DataFrame],
) -> None:
    """Write TensorBoard summary for the current step observations."""
    for tag in models:
        if tag not in tagged_output:
            continue

        with tf.name_scope(f"{tag}.model"):
            models[tag].log(datasets[tag])

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

            hue_order = ["initial", "old", "new"]
            palette = {"initial": "tab:green", "old": "tab:green", "new": "tab:orange"}
            markers = {"initial": "X", "old": "o", "new": "o"}

            # assume that any OBJECTIVE- or single-tagged multi-output dataset => multi-objective
            # more complex scenarios (e.g. constrained data) need to be plotted by the acq function
            if len(datasets) > 1 and tag != OBJECTIVE:
                observation_plot_df["observation type"] = observation_plot_df.apply(
                    lambda x: x["observations"],
                    axis=1,
                )
            else:
                observation_plot_df["pareto"] = non_dominated(datasets[tag].observations)[1]
                observation_plot_df["observation type"] = observation_plot_df.apply(
                    lambda x: x["observations"] + x["pareto"] * " (non-dominated)",
                    axis=1,
                )
                hue_order += [hue + " (non-dominated)" for hue in hue_order]
                palette.update(
                    {
                        "initial (non-dominated)": "tab:purple",
                        "old (non-dominated)": "tab:purple",
                        "new (non-dominated)": "tab:red",
                    }
                )
                markers.update(
                    {
                        "initial (non-dominated)": "X",
                        "old (non-dominated)": "o",
                        "new (non-dominated)": "o",
                    }
                )

            pairplot = sns.pairplot(
                observation_plot_df,
                vars=columns,
                hue="observation type",
                hue_order=hue_order,
                palette=palette,
                markers=markers,
            )
            logging.pyplot(f"{tag}.observations/_pairplot", pairplot.fig)
            observation_plot_df.loc[
                observation_plot_df["observations"] == "new", "observations"
            ] = "old"
            observation_plot_dfs[tag] = observation_plot_df

    logging.scalar(
        "wallclock/model_fitting",
        model_fitting_timer.time,
    )


def write_summary_query_points(
    datasets: Mapping[Tag, Dataset],
    models: Mapping[Tag, ProbabilisticModel],
    search_space: SearchSpace,
    query_points: TensorType,
    query_point_generation_timer: Timer,
    query_plot_dfs: MutableMapping[int, pd.DataFrame],
) -> None:
    """Write TensorBoard summary for the current step query points."""

    if tf.rank(query_points) == 2:
        for i in tf.range(tf.shape(query_points)[1]):
            if len(query_points) == 1:
                logging.scalar(f"query_point/[{i}]", float(query_points[0, i]))
            else:
                logging.histogram(f"query_points/[{i}]", query_points[:, i])
        logging.histogram("query_points/euclidean_distances", lambda: pdist(query_points))

    if pd and sns and logging.include_summary("query_points/_pairplot"):
        columns = [f"x{i}" for i in range(tf.shape(query_points)[1])]
        qp_preds = query_points
        for tag in datasets:
            pred = models[tag].predict(query_points)[0]
            qp_preds = tf.concat([qp_preds, tf.cast(pred, query_points.dtype)], 1)
            output_dim = tf.shape(pred)[-1]
            for i in range(output_dim):
                columns.append(f"{tag}{i if (output_dim > 1) else ''} predicted")
        query_new_df = pd.DataFrame(qp_preds, columns=columns).applymap(float)
        query_new_df["query points"] = "new"
        query_plot_df = pd.concat(
            (query_plot_dfs.get(0), query_new_df), copy=False, ignore_index=True
        )
        pairplot = sns.pairplot(
            query_plot_df, hue="query points", hue_order=["old", "new"], height=2.25
        )
        padding = 0.025 * (search_space.upper - search_space.lower)
        upper_limits = search_space.upper + padding
        lower_limits = search_space.lower - padding
        for i in range(search_space.dimension):
            pairplot.axes[0, i].set_xlim((lower_limits[i], upper_limits[i]))
            pairplot.axes[i, 0].set_ylim((lower_limits[i], upper_limits[i]))
        logging.pyplot("query_points/_pairplot", pairplot.fig)
        query_plot_df["query points"] = "old"
        query_plot_dfs[0] = query_plot_df

    logging.scalar(
        "wallclock/query_point_generation",
        query_point_generation_timer.time,
    )


def stop_at_minimum(
    minimum: Optional[tf.Tensor] = None,
    minimizers: Optional[tf.Tensor] = None,
    minimum_atol: float = 0,
    minimum_rtol: float = 0.05,
    minimizers_atol: float = 0,
    minimizers_rtol: float = 0.05,
    objective_tag: Tag = OBJECTIVE,
    minimum_step_number: Optional[int] = None,
) -> EarlyStopCallback[TrainableProbabilisticModel, object]:
    """
    Generate an early stop function that terminates a BO loop when it gets close enough to the
    given objective minimum and/or minimizer points.

    :param minimum: Optional minimum to stop at, with shape [1].
    :param minimizers: Optional minimizer points to stop at, with shape [N, D].
    :param minimum_atol: Absolute tolerance for minimum.
    :param minimum_rtol: Relative tolerance for minimum.
    :param minimizers_atol: Absolute tolerance for minimizer point.
    :param minimizers_rtol: Relative tolerance for minimizer point.
    :param objective_tag: The tag for the objective data.
    :param minimum_step_number: Minimum step number to stop at.
    :return: An early stop function that terminates if we get close enough to both the minimum
        and any of the minimizer points.
    """

    def early_stop_callback(
        datasets: Mapping[Tag, Dataset],
        _models: Mapping[Tag, TrainableProbabilisticModel],
        _acquisition_state: object,
    ) -> bool:
        if minimum_step_number is not None and logging.get_step_number() < minimum_step_number:
            return False
        dataset = datasets[objective_tag]
        arg_min_idx = tf.squeeze(tf.argmin(dataset.observations, axis=0))
        if minimum is not None:
            best_y = dataset.observations[arg_min_idx]
            close_y = np.isclose(best_y, minimum, atol=minimum_atol, rtol=minimum_rtol)
            if not tf.reduce_all(close_y):
                return False
        if minimizers is not None:
            best_x = dataset.query_points[arg_min_idx]
            close_x = np.isclose(best_x, minimizers, atol=minimizers_atol, rtol=minimizers_rtol)
            if not tf.reduce_any(tf.reduce_all(close_x, axis=-1), axis=0):
                return False
        return True

    return early_stop_callback
