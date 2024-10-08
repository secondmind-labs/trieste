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
This module contains the Ask/Tell API for users of Trieste who would like to
perform Bayesian Optimization with external control of the optimization loop.
"""


from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Generic, Mapping, Optional, Sequence, Type, TypeVar, cast, overload

import tensorflow as tf

from .models.utils import optimize_model_and_save_result

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None

from . import logging
from .acquisition.rule import (
    AcquisitionRule,
    EfficientGlobalOptimization,
    LocalDatasetsAcquisitionRule,
)
from .acquisition.utils import with_local_datasets
from .bayesian_optimizer import (
    FrozenRecord,
    OptimizationResult,
    Record,
    observation_plot_init,
    write_summary_initial_model_fit,
    write_summary_observations,
    write_summary_query_points,
)
from .data import Dataset
from .models import ProbabilisticModel, TrainableProbabilisticModel
from .observer import OBJECTIVE
from .space import SearchSpace
from .types import State, Tag, TensorType
from .utils import Ok, Timer
from .utils.misc import LocalizedTag, get_value_for_tag, ignoring_local_tags

StateType = TypeVar("StateType")
""" Unbound type variable. """

SearchSpaceType = TypeVar("SearchSpaceType", bound=SearchSpace)
""" Type variable bound to :class:`SearchSpace`. """

ProbabilisticModelType = TypeVar(
    "ProbabilisticModelType", bound=ProbabilisticModel, contravariant=True
)
""" Contravariant type variable bound to :class:`ProbabilisticModel`. """

AskTellOptimizerType = TypeVar("AskTellOptimizerType")


@dataclass(frozen=True)
class AskTellOptimizerState(Generic[StateType, ProbabilisticModelType]):
    """
    Internal state for an Ask/Tell optimizer. This can be obtained using the optimizer's
    `to_state` method, and can be used to initialise a new instance of the optimizer.
    """

    record: Record[StateType, ProbabilisticModelType]
    """ A record of the current state of the optimization. """

    local_data_ixs: Optional[Sequence[TensorType]]
    """ Indices to the local data, for LocalDatasetsAcquisitionRule rules
    when `track_data` is `False`. """

    local_data_len: Optional[int]
    """ Length of the datasets, for LocalDatasetsAcquisitionRule rules
    when `track_data` is `False`. """


class AskTellOptimizerABC(ABC, Generic[SearchSpaceType, ProbabilisticModelType]):
    """
    This class provides Ask/Tell optimization interface. It is designed for those use cases
    when control of the optimization loop by Trieste is impossible or not desirable.
    For the default use case with model training, refer to :class:`AskTellOptimizer`.
    For more details about the Bayesian Optimization routine, refer to :class:`BayesianOptimizer`.
    """

    @overload
    def __init__(
        self,
        search_space: SearchSpaceType,
        datasets: Mapping[Tag, Dataset],
        models: Mapping[Tag, ProbabilisticModelType],
        *,
        fit_model: bool = True,
        track_data: bool = True,
        local_data_ixs: Optional[Sequence[TensorType]] = None,
        local_data_len: Optional[int] = None,
    ): ...

    @overload
    def __init__(
        self,
        search_space: SearchSpaceType,
        datasets: Mapping[Tag, Dataset],
        models: Mapping[Tag, ProbabilisticModelType],
        acquisition_rule: AcquisitionRule[TensorType, SearchSpaceType, ProbabilisticModelType],
        *,
        fit_model: bool = True,
        track_data: bool = True,
        local_data_ixs: Optional[Sequence[TensorType]] = None,
        local_data_len: Optional[int] = None,
    ): ...

    @overload
    def __init__(
        self,
        search_space: SearchSpaceType,
        datasets: Mapping[Tag, Dataset],
        models: Mapping[Tag, ProbabilisticModelType],
        acquisition_rule: AcquisitionRule[
            State[StateType | None, TensorType], SearchSpaceType, ProbabilisticModelType
        ],
        acquisition_state: StateType | None,
        *,
        fit_model: bool = True,
        track_data: bool = True,
        local_data_ixs: Optional[Sequence[TensorType]] = None,
        local_data_len: Optional[int] = None,
    ): ...

    @overload
    def __init__(
        self,
        search_space: SearchSpaceType,
        datasets: Dataset,
        models: ProbabilisticModelType,
        *,
        fit_model: bool = True,
        track_data: bool = True,
        local_data_ixs: Optional[Sequence[TensorType]] = None,
        local_data_len: Optional[int] = None,
    ): ...

    @overload
    def __init__(
        self,
        search_space: SearchSpaceType,
        datasets: Dataset,
        models: ProbabilisticModelType,
        acquisition_rule: AcquisitionRule[TensorType, SearchSpaceType, ProbabilisticModelType],
        *,
        fit_model: bool = True,
        track_data: bool = True,
        local_data_ixs: Optional[Sequence[TensorType]] = None,
        local_data_len: Optional[int] = None,
    ): ...

    @overload
    def __init__(
        self,
        search_space: SearchSpaceType,
        datasets: Dataset,
        models: ProbabilisticModelType,
        acquisition_rule: AcquisitionRule[
            State[StateType | None, TensorType], SearchSpaceType, ProbabilisticModelType
        ],
        acquisition_state: StateType | None = None,
        *,
        fit_model: bool = True,
        track_data: bool = True,
        local_data_ixs: Optional[Sequence[TensorType]] = None,
        local_data_len: Optional[int] = None,
    ): ...

    def __init__(
        self,
        search_space: SearchSpaceType,
        datasets: Mapping[Tag, Dataset] | Dataset,
        models: Mapping[Tag, ProbabilisticModelType] | ProbabilisticModelType,
        acquisition_rule: (
            AcquisitionRule[
                TensorType | State[StateType | None, TensorType],
                SearchSpaceType,
                ProbabilisticModelType,
            ]
            | None
        ) = None,
        acquisition_state: StateType | None = None,
        *,
        fit_model: bool = True,
        track_data: bool = True,
        local_data_ixs: Optional[Sequence[TensorType]] = None,
        local_data_len: Optional[int] = None,
    ):
        """
        :param search_space: The space over which to search for the next query point.
        :param datasets: Already observed input-output pairs for each tag.
        :param models: The model to use for each :class:`~trieste.data.Dataset` in
            ``datasets``.
        :param acquisition_rule: The acquisition rule, which defines how to search for a new point
            on each optimization step. Defaults to
            :class:`~trieste.acquisition.rule.EfficientGlobalOptimization` with default
            arguments. Note that if the default is used, this implies the tags must be
            `OBJECTIVE` and the search space can be any :class:`~trieste.space.SearchSpace`.
        :param acquisition_state: The optional acquisition state for stateful acquisitions.
        :param fit_model: If `True` (default), models passed in will be optimized on the given data.
            If `False`, the models are assumed to be optimized already.
        :param track_data: If `True` (default), the optimizer will track the changing
            datasets via a local copy. If `False`, it will infer new datasets from
            updates to the global datasets (optionally using `local_data_ixs` and indices passed
            in to `tell`).
        :param local_data_ixs: Indices to the local data in the initial datasets. If unspecified,
            assumes that the initial datasets are global.
        :param local_data_len: Optional length of the data when the passed in `local_data_ixs`
            were measured. If the data has increased since then, the indices are extended.
        :raise ValueError: If any of the following are true:
            - the keys in ``datasets`` and ``models`` do not match
            - ``datasets`` or ``models`` are empty
            - default acquisition is used but incompatible with other inputs
        """
        self._search_space = search_space
        self._acquisition_record = self._acquisition_state = acquisition_state

        if not datasets or not models:
            raise ValueError("dicts of datasets and models must be populated.")

        if isinstance(datasets, Dataset):
            datasets = {OBJECTIVE: datasets}
        if not isinstance(models, Mapping):
            models = {OBJECTIVE: models}

        # reassure the type checker that everything is tagged
        datasets = cast(Dict[Tag, Dataset], datasets)
        models = cast(Dict[Tag, ProbabilisticModelType], models)

        # Get set of dataset and model keys, ignoring any local tag index. That is, only the
        # global tag part is considered.
        datasets_keys = {LocalizedTag.from_tag(tag).global_tag for tag in datasets.keys()}
        models_keys = {LocalizedTag.from_tag(tag).global_tag for tag in models.keys()}
        if datasets_keys != models_keys:
            raise ValueError(
                f"datasets and models should contain the same keys. Got {datasets_keys} and"
                f" {models_keys} respectively."
            )

        self._datasets = datasets
        self._models = models
        self.track_data = track_data

        self._query_plot_dfs: dict[int, pd.DataFrame] = {}
        self._observation_plot_dfs = observation_plot_init(self._datasets)

        if acquisition_rule is None:
            if self._datasets.keys() != {OBJECTIVE}:
                raise ValueError(
                    f"Default acquisition rule EfficientGlobalOptimization requires tag"
                    f" {OBJECTIVE!r}, got keys {self._datasets.keys()}"
                )

            self._acquisition_rule = cast(
                AcquisitionRule[TensorType, SearchSpaceType, ProbabilisticModelType],
                EfficientGlobalOptimization(),
            )
        else:
            self._acquisition_rule = acquisition_rule

        if isinstance(self._acquisition_rule, LocalDatasetsAcquisitionRule):
            # In order to support local datasets, account for the case where there may be an initial
            # dataset that is not tagged per region. In this case, only the global dataset will
            # exist in datasets. We want to copy this initial dataset to all the regions.
            num_local_datasets = self._acquisition_rule.num_local_datasets
            if self.track_data:
                datasets = self._datasets = with_local_datasets(self._datasets, num_local_datasets)
            else:
                self._dataset_len = self.dataset_len(self._datasets)
                if local_data_ixs is not None:
                    self._dataset_ixs = list(local_data_ixs)
                    if local_data_len is not None:
                        # infer new dataset indices from change in dataset sizes
                        num_new_points = self._dataset_len - local_data_len
                        if num_new_points < 0 or (
                            num_local_datasets > 0 and num_new_points % num_local_datasets != 0
                        ):
                            raise ValueError(
                                "Cannot infer new data points as datasets haven't increased by "
                                f"a multiple of {num_local_datasets}"
                            )
                        for i in range(num_local_datasets):
                            self._dataset_ixs[i] = tf.concat(
                                [
                                    self._dataset_ixs[i],
                                    tf.range(0, num_new_points, num_local_datasets)
                                    + local_data_len
                                    + i,
                                ],
                                -1,
                            )
                else:
                    self._dataset_ixs = [
                        tf.range(self._dataset_len) for _ in range(num_local_datasets)
                    ]

                datasets = with_local_datasets(
                    self._datasets, num_local_datasets, self._dataset_ixs
                )
            self._acquisition_rule.initialize_subspaces(search_space)

        filtered_datasets: (
            Mapping[Tag, Dataset] | State[StateType | None, Mapping[Tag, Dataset]]
        ) = self._acquisition_rule.filter_datasets(self._models, datasets)
        if callable(filtered_datasets):
            self._acquisition_state, self._filtered_datasets = filtered_datasets(
                self._acquisition_state
            )
        else:
            self._filtered_datasets = filtered_datasets

        if fit_model:
            with Timer() as initial_model_fitting_timer:
                for tag, model in self._models.items():
                    # Prefer local dataset if available.
                    tags = [tag, LocalizedTag.from_tag(tag).global_tag]
                    _, dataset = get_value_for_tag(self._filtered_datasets, *tags)
                    assert dataset is not None
                    self.update_model(model, dataset)

            summary_writer = logging.get_tensorboard_writer()
            if summary_writer:
                with summary_writer.as_default(step=logging.get_step_number()):
                    write_summary_initial_model_fit(
                        self._datasets, self._models, initial_model_fitting_timer
                    )

    @abstractmethod
    def update_model(self, model: ProbabilisticModelType, dataset: Dataset) -> None:
        """
        Update the model on the specified dataset, for example by training.
        Called during the Tell stage and optionally at initial fitting.
        """

    def __repr__(self) -> str:
        """Print-friendly string representation"""
        return f"""AskTellOptimizer({self._search_space!r}, {self._datasets!r},
               {self._models!r}, {self._acquisition_rule!r}), "
               {self._acquisition_state!r}"""

    @property
    def datasets(self) -> Mapping[Tag, Dataset]:
        """The current datasets."""
        return self._datasets

    @property
    def dataset(self) -> Dataset:
        """The current dataset when there is just one dataset."""
        # Ignore local datasets.
        datasets: Mapping[Tag, Dataset] = ignoring_local_tags(self.datasets)
        if len(datasets) == 1:
            return next(iter(datasets.values()))
        else:
            raise ValueError(f"Expected a single dataset, found {len(datasets)}")

    @property
    def local_data_ixs(self) -> Optional[Sequence[TensorType]]:
        """Indices to the local data. Only stored for LocalDatasetsAcquisitionRule rules
        when `track_data` is `False`."""
        if isinstance(self._acquisition_rule, LocalDatasetsAcquisitionRule) and not self.track_data:
            return self._dataset_ixs
        return None

    @property
    def local_data_len(self) -> Optional[int]:
        """Data length. Only stored for LocalDatasetsAcquisitionRule rules
        when `track_data` is `False`."""
        if isinstance(self._acquisition_rule, LocalDatasetsAcquisitionRule) and not self.track_data:
            return self._dataset_len
        return None

    @property
    def models(self) -> Mapping[Tag, ProbabilisticModelType]:
        """The current models."""
        return self._models

    @models.setter
    def models(self, models: Mapping[Tag, ProbabilisticModelType]) -> None:
        """Update the current models."""
        if models.keys() != self.models.keys():
            raise ValueError(
                f"New models contain incorrect keys. Expected {self.models.keys()}, "
                f"received {models.keys()}."
            )
        self._models = dict(models)

    @property
    def model(self) -> ProbabilisticModel:
        """The current model when there is just one model."""
        # Ignore local models.
        models: Mapping[Tag, ProbabilisticModel] = ignoring_local_tags(self.models)
        if len(models) == 1:
            return next(iter(models.values()))
        else:
            raise ValueError(f"Expected a single model, found {len(models)}")

    @model.setter
    def model(self, model: ProbabilisticModelType) -> None:
        """Update the current model, using the OBJECTIVE tag."""
        if len(self.models) != 1:
            raise ValueError(f"Expected a single model, found {len(self.models)}")
        if self.models.keys() != {OBJECTIVE}:
            raise ValueError(
                f"Expected a single model tagged OBJECTIVE, found {self.models.keys()}. "
                "To update this, pass in a dictionary to the models property instead."
            )
        self._models = {OBJECTIVE: model}

    @property
    def acquisition_state(self) -> StateType | None:
        """The current acquisition state."""
        return self._acquisition_state

    @classmethod
    def dataset_len(cls, datasets: Mapping[Tag, Dataset]) -> int:
        """Helper method for inferring the global dataset size."""
        dataset_lens = {
            tag: int(tf.shape(dataset.query_points)[0])
            for tag, dataset in datasets.items()
            if not LocalizedTag.from_tag(tag).is_local
        }
        unique_lens, _ = tf.unique(list(dataset_lens.values()))
        if len(unique_lens) == 1:
            return int(unique_lens[0])
        else:
            raise ValueError(
                f"Expected unique global dataset size, got {unique_lens}: {dataset_lens}"
            )

    @classmethod
    def from_record(
        cls: Type[AskTellOptimizerType],
        record: (
            Record[StateType, ProbabilisticModelType]
            | FrozenRecord[StateType, ProbabilisticModelType]
        ),
        search_space: SearchSpaceType,
        acquisition_rule: (
            AcquisitionRule[
                TensorType | State[StateType | None, TensorType],
                SearchSpaceType,
                ProbabilisticModelType,
            ]
            | None
        ) = None,
        track_data: bool = True,
        local_data_ixs: Optional[Sequence[TensorType]] = None,
        local_data_len: Optional[int] = None,
    ) -> AskTellOptimizerType:
        """Creates new :class:`~AskTellOptimizer` instance from provided optimization state.
        Model training isn't triggered upon creation of the instance.

        :param record: Optimization state record.
        :param search_space: The space over which to search for the next query point.
        :param acquisition_rule: The acquisition rule, which defines how to search for a new point
            on each optimization step. Defaults to
            :class:`~trieste.acquisition.rule.EfficientGlobalOptimization` with default
            arguments.
        :param track_data: Whether the optimizer tracks the changing datasets via a local copy.
        :param local_data_ixs: Indices to local data for local rules with `track_data` False.
        :param local_data_len: Original data length for local rules with `track_data` False.
        :return: New instance of :class:`~AskTellOptimizer`.
        """
        # we are recovering previously saved optimization state
        # so the model was already trained
        # thus there is no need to train it again

        # type ignore below is because this relies on subclasses not overriding __init__
        # ones that do may also need to override this to get it to work
        return cls(  # type: ignore
            search_space,
            record.datasets,
            record.models,
            acquisition_rule=acquisition_rule,
            acquisition_state=record.acquisition_state,
            fit_model=False,
            track_data=track_data,
            local_data_ixs=local_data_ixs,
            local_data_len=local_data_len,
        )

    def to_record(self, copy: bool = True) -> Record[StateType, ProbabilisticModelType]:
        """Collects the current state of the optimization, which includes datasets,
        models and acquisition state (if applicable).

        :param copy: Whether to return a copy of the current state or the original. Copying
            is not supported for all model types. However, continuing the optimization will
            modify the original state.
        :return: An optimization state record.
        """
        try:
            datasets_copy = deepcopy(self._datasets) if copy else self._datasets
            models_copy = deepcopy(self._models) if copy else self._models
            # use the state as it was at acquisition time, not the one modified in
            # filter_datasets in preparation for the next acquisition, so we can reinitialise
            # the AskTellOptimizer using the record
            state_copy = deepcopy(self._acquisition_record) if copy else self._acquisition_record
        except Exception as e:
            raise NotImplementedError(
                "Failed to copy the optimization state. Some models do not support "
                "deecopying (this is particularly common for deep neural network models). "
                "For these models, the `copy` argument of the `to_record` or `to_result` "
                "methods should be set to `False`. This means that the returned state may be "
                "modified by subsequent optimization."
            ) from e

        return Record(datasets=datasets_copy, models=models_copy, acquisition_state=state_copy)

    def to_result(self, copy: bool = True) -> OptimizationResult[StateType, ProbabilisticModelType]:
        """Converts current state of the optimization
        into a :class:`~trieste.data.OptimizationResult` object.

        :param copy: Whether to return a copy of the current state or the original. Copying
            is not supported for all model types. However, continuing the optimization will
            modify the original state.
        :return: A :class:`~trieste.data.OptimizationResult` object.
        """
        record: Record[StateType, ProbabilisticModelType] = self.to_record(copy=copy)
        return OptimizationResult(Ok(record), [])

    @classmethod
    def from_state(
        cls: Type[AskTellOptimizerType],
        state: AskTellOptimizerState[StateType, ProbabilisticModelType],
        search_space: SearchSpaceType,
        acquisition_rule: (
            AcquisitionRule[
                TensorType | State[StateType | None, TensorType],
                SearchSpaceType,
                ProbabilisticModelType,
            ]
            | None
        ) = None,
        track_data: bool = True,
    ) -> AskTellOptimizerType:
        """Creates new :class:`~AskTellOptimizer` instance from provided AskTellOptimizer state.
        Model training isn't triggered upon creation of the instance.

        :param state: AskTellOptimizer state.
        :param search_space: The space over which to search for the next query point.
        :param acquisition_rule: The acquisition rule, which defines how to search for a new point
            on each optimization step. Defaults to
            :class:`~trieste.acquisition.rule.EfficientGlobalOptimization` with default
            arguments.
        :param track_data: Whether the optimizer tracks the changing datasets via a local copy.
        :return: New instance of :class:`~AskTellOptimizer`.
        """
        return cls.from_record(  # type: ignore
            state.record,
            search_space,
            acquisition_rule,
            track_data=track_data,
            local_data_ixs=state.local_data_ixs,
            local_data_len=state.local_data_len,
        )

    def to_state(
        self, copy: bool = False
    ) -> AskTellOptimizerState[StateType, ProbabilisticModelType]:
        """Returns the AskTellOptimizer state, comprising the current optimization state
        alongside any internal AskTellOptimizer state.

        :param copy: Whether to return a copy of the current state or the original. Copying
            is not supported for all model types. However, continuing the optimization will
            modify the original state.
        :return: An :class:`AskTellOptimizerState` object.
        """
        return AskTellOptimizerState(
            record=self.to_record(copy=copy),
            local_data_ixs=self.local_data_ixs,
            local_data_len=self.local_data_len,
        )

    def ask(self) -> TensorType:
        """Suggests a point (or points in batch mode) to observe by optimizing the acquisition
        function. If the acquisition is stateful, its state is saved.

        :return: A :class:`TensorType` instance representing suggested point(s).
        """
        # This trick deserves a comment to explain what's going on
        # acquisition_rule.acquire can return different things:
        # - when acquisition has no state attached, it returns just points
        # - when acquisition has state, it returns a Callable
        #   which, when called, returns state and points
        # so code below is needed to cater for both cases

        with Timer() as query_point_generation_timer:
            points_or_stateful = self._acquisition_rule.acquire(
                self._search_space, self._models, datasets=self._filtered_datasets
            )

        if callable(points_or_stateful):
            self._acquisition_state, query_points = points_or_stateful(self._acquisition_state)
            # also keep a copy of the state to return in to_record
            self._acquisition_record = self._acquisition_state
        else:
            query_points = points_or_stateful

        summary_writer = logging.get_tensorboard_writer()
        if summary_writer:
            with summary_writer.as_default(step=logging.get_step_number()):
                write_summary_query_points(
                    self._datasets,
                    self._models,
                    self._search_space,
                    query_points,
                    query_point_generation_timer,
                    self._query_plot_dfs,
                )

        return query_points

    def tell(
        self,
        new_data: Mapping[Tag, Dataset] | Dataset,
        new_data_ixs: Optional[Sequence[TensorType]] = None,
    ) -> None:
        """Updates optimizer state with new data.

        :param new_data: New observed data. If `track_data` is `False`, this refers to all
            the data.
        :param new_data_ixs: Indices to the new observed local data, if `track_data` is `False`.
            If unspecified, inferred from the change in dataset sizes.
        :raise ValueError: If keys in ``new_data`` do not match those in already built dataset.
        """
        if isinstance(new_data, Dataset):
            new_data = {OBJECTIVE: new_data}

        # The datasets must have the same keys as the existing datasets. Only exception is if
        # the existing datasets are all global and the new data contains local datasets too.
        if all(LocalizedTag.from_tag(tag).local_index is None for tag in self._datasets.keys()):
            global_old = {LocalizedTag.from_tag(tag).global_tag for tag in self._datasets.keys()}
            global_new = {LocalizedTag.from_tag(tag).global_tag for tag in new_data.keys()}
            if global_new != global_old:
                raise ValueError(
                    f"new_data global keys {global_new} doesn't "
                    f"match dataset global keys {global_old}"
                )
        elif self._datasets.keys() != new_data.keys():
            raise ValueError(
                f"new_data keys {new_data.keys()} doesn't "
                f"match dataset keys {self._datasets.keys()}"
            )

        if self.track_data:
            for tag, new_dataset in new_data.items():
                self._datasets[tag] += new_dataset
            datasets: Mapping[Tag, Dataset] = self._datasets
        elif not isinstance(self._acquisition_rule, LocalDatasetsAcquisitionRule):
            datasets = new_data
        else:
            num_local_datasets = len(self._dataset_ixs)
            if new_data_ixs is None:
                # infer dataset indices from change in dataset sizes
                new_dataset_len = self.dataset_len(new_data)
                num_new_points = new_dataset_len - self._dataset_len
                if num_new_points < 0 or (
                    num_local_datasets > 0 and num_new_points % num_local_datasets != 0
                ):
                    raise ValueError(
                        "Cannot infer new data points as datasets haven't increased by "
                        f"a multiple of {num_local_datasets}"
                    )
                for i in range(num_local_datasets):
                    self._dataset_ixs[i] = tf.concat(
                        [
                            self._dataset_ixs[i],
                            tf.range(0, num_new_points, num_local_datasets) + self._dataset_len + i,
                        ],
                        -1,
                    )
            else:
                # use explicit indices
                if len(new_data_ixs) != num_local_datasets:
                    raise ValueError(
                        f"new_data_ixs has {len(new_data_ixs)} entries, "
                        f"expected {num_local_datasets}"
                    )
                for i in range(num_local_datasets):
                    self._dataset_ixs[i] = tf.concat([self._dataset_ixs[i], new_data_ixs[i]], -1)
            datasets = with_local_datasets(new_data, num_local_datasets, self._dataset_ixs)
            self._dataset_len = self.dataset_len(datasets)

        filtered_datasets = self._acquisition_rule.filter_datasets(self._models, datasets)
        if callable(filtered_datasets):
            self._acquisition_state, self._filtered_datasets = filtered_datasets(
                self._acquisition_state
            )
        else:
            self._filtered_datasets = filtered_datasets

        with Timer() as model_fitting_timer:
            for tag, model in self._models.items():
                # Always use the matching dataset to the model. If the model is
                # local, then the dataset should be too by this stage.
                dataset = self._filtered_datasets[tag]
                self.update_model(model, dataset)

        summary_writer = logging.get_tensorboard_writer()
        if summary_writer:
            with summary_writer.as_default(step=logging.get_step_number()):
                write_summary_observations(
                    datasets,
                    self._models,
                    new_data,
                    model_fitting_timer,
                    self._observation_plot_dfs,
                )


TrainableProbabilisticModelType = TypeVar(
    "TrainableProbabilisticModelType", bound=TrainableProbabilisticModel, contravariant=True
)
""" Contravariant type variable bound to :class:`TrainableProbabilisticModel`. """


class AskTellOptimizer(AskTellOptimizerABC[SearchSpaceType, TrainableProbabilisticModelType]):
    """
    This class provides Ask/Tell optimization interface with the default model training
    using the TrainableProbabilisticModel interface.
    """

    def update_model(self, model: TrainableProbabilisticModelType, dataset: Dataset) -> None:
        model.update(dataset)
        optimize_model_and_save_result(model, dataset)


class AskTellOptimizerNoTraining(AskTellOptimizerABC[SearchSpaceType, ProbabilisticModelType]):
    """
    This class provides Ask/Tell optimization interface with no model training performed
    during the Tell stage or at initialization.
    """

    def update_model(self, model: ProbabilisticModelType, dataset: Dataset) -> None:
        pass
