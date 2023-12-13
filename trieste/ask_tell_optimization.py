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

from copy import deepcopy
from typing import Dict, Generic, Mapping, TypeVar, cast, overload

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
from .models import TrainableProbabilisticModel
from .observer import OBJECTIVE
from .space import SearchSpace
from .types import State, Tag, TensorType
from .utils import Ok, Timer
from .utils.misc import LocalizedTag, get_value_for_tag, ignoring_local_tags

StateType = TypeVar("StateType")
""" Unbound type variable. """

SearchSpaceType = TypeVar("SearchSpaceType", bound=SearchSpace)
""" Type variable bound to :class:`SearchSpace`. """

TrainableProbabilisticModelType = TypeVar(
    "TrainableProbabilisticModelType", bound=TrainableProbabilisticModel, contravariant=True
)
""" Contravariant type variable bound to :class:`TrainableProbabilisticModel`. """


class AskTellOptimizer(Generic[SearchSpaceType, TrainableProbabilisticModelType]):
    """
    This class provides Ask/Tell optimization interface. It is designed for those use cases
    when control of the optimization loop by Trieste is impossible or not desirable.
    For more details about the Bayesian Optimization routine, refer to :class:`BayesianOptimizer`.
    """

    @overload
    def __init__(
        self,
        search_space: SearchSpaceType,
        datasets: Mapping[Tag, Dataset],
        models: Mapping[Tag, TrainableProbabilisticModelType],
        *,
        fit_model: bool = True,
    ):
        ...

    @overload
    def __init__(
        self,
        search_space: SearchSpaceType,
        datasets: Mapping[Tag, Dataset],
        models: Mapping[Tag, TrainableProbabilisticModelType],
        acquisition_rule: AcquisitionRule[
            TensorType, SearchSpaceType, TrainableProbabilisticModelType
        ],
        *,
        fit_model: bool = True,
    ):
        ...

    @overload
    def __init__(
        self,
        search_space: SearchSpaceType,
        datasets: Mapping[Tag, Dataset],
        models: Mapping[Tag, TrainableProbabilisticModelType],
        acquisition_rule: AcquisitionRule[
            State[StateType | None, TensorType], SearchSpaceType, TrainableProbabilisticModelType
        ],
        acquisition_state: StateType | None,
        *,
        fit_model: bool = True,
    ):
        ...

    @overload
    def __init__(
        self,
        search_space: SearchSpaceType,
        datasets: Dataset,
        models: TrainableProbabilisticModelType,
        *,
        fit_model: bool = True,
    ):
        ...

    @overload
    def __init__(
        self,
        search_space: SearchSpaceType,
        datasets: Dataset,
        models: TrainableProbabilisticModelType,
        acquisition_rule: AcquisitionRule[
            TensorType, SearchSpaceType, TrainableProbabilisticModelType
        ],
        *,
        fit_model: bool = True,
    ):
        ...

    @overload
    def __init__(
        self,
        search_space: SearchSpaceType,
        datasets: Dataset,
        models: TrainableProbabilisticModelType,
        acquisition_rule: AcquisitionRule[
            State[StateType | None, TensorType], SearchSpaceType, TrainableProbabilisticModelType
        ],
        acquisition_state: StateType | None = None,
        *,
        fit_model: bool = True,
    ):
        ...

    def __init__(
        self,
        search_space: SearchSpaceType,
        datasets: Mapping[Tag, Dataset] | Dataset,
        models: Mapping[Tag, TrainableProbabilisticModelType] | TrainableProbabilisticModelType,
        acquisition_rule: AcquisitionRule[
            TensorType | State[StateType | None, TensorType],
            SearchSpaceType,
            TrainableProbabilisticModelType,
        ]
        | None = None,
        acquisition_state: StateType | None = None,
        *,
        fit_model: bool = True,
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
        :raise ValueError: If any of the following are true:
            - the keys in ``datasets`` and ``models`` do not match
            - ``datasets`` or ``models`` are empty
            - default acquisition is used but incompatible with other inputs
        """
        self._search_space = search_space
        self._acquisition_state = acquisition_state

        if not datasets or not models:
            raise ValueError("dicts of datasets and models must be populated.")

        # Copy the dataset so we don't change the one provided by the user.
        datasets = deepcopy(datasets)

        if isinstance(datasets, Dataset):
            datasets = {OBJECTIVE: datasets}
        if not isinstance(models, Mapping):
            models = {OBJECTIVE: models}

        # reassure the type checker that everything is tagged
        datasets = cast(Dict[Tag, Dataset], datasets)
        models = cast(Dict[Tag, TrainableProbabilisticModelType], models)

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

        self._query_plot_dfs: dict[int, pd.DataFrame] = {}
        self._observation_plot_dfs = observation_plot_init(self._datasets)

        if acquisition_rule is None:
            if self._datasets.keys() != {OBJECTIVE}:
                raise ValueError(
                    f"Default acquisition rule EfficientGlobalOptimization requires tag"
                    f" {OBJECTIVE!r}, got keys {self._datasets.keys()}"
                )

            self._acquisition_rule = cast(
                AcquisitionRule[TensorType, SearchSpaceType, TrainableProbabilisticModelType],
                EfficientGlobalOptimization(),
            )
        else:
            self._acquisition_rule = acquisition_rule

        # In order to support local datasets, account for the case where there may be an initial
        # dataset that is not tagged per region. In this case, only the global dataset will
        # exist in datasets. We want to copy this initial dataset to all the regions.
        # Copy the global dataset if the local version for the subspace is not available.
        #
        # Only applies to a subset of acquisition rules, i.e. ones that have subspaces and
        # hence use local datasets.
        if isinstance(self._acquisition_rule, LocalDatasetsAcquisitionRule):
            self._datasets = with_local_datasets(
                self._datasets, self._acquisition_rule.num_local_datasets
            )
        self._filtered_datasets = self._acquisition_rule.filter_datasets(
            self._models, self._datasets
        )

        if fit_model:
            with Timer() as initial_model_fitting_timer:
                for tag, model in self._models.items():
                    # Prefer local dataset if available.
                    tags = [tag, LocalizedTag.from_tag(tag).global_tag]
                    _, dataset = get_value_for_tag(self._filtered_datasets, *tags)
                    assert dataset is not None
                    model.update(dataset)
                    optimize_model_and_save_result(model, dataset)

            summary_writer = logging.get_tensorboard_writer()
            if summary_writer:
                with summary_writer.as_default(step=logging.get_step_number()):
                    write_summary_initial_model_fit(
                        self._datasets, self._models, initial_model_fitting_timer
                    )

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
    def models(self) -> Mapping[Tag, TrainableProbabilisticModelType]:
        """The current models."""
        return self._models

    @models.setter
    def models(self, models: Mapping[Tag, TrainableProbabilisticModelType]) -> None:
        """Update the current models."""
        if models.keys() != self.models.keys():
            raise ValueError(
                f"New models contain incorrect keys. Expected {self.models.keys()}, "
                f"received {models.keys()}."
            )
        self._models = dict(models)

    @property
    def model(self) -> TrainableProbabilisticModel:
        """The current model when there is just one model."""
        # Ignore local models.
        models: Mapping[Tag, TrainableProbabilisticModel] = ignoring_local_tags(self.models)
        if len(models) == 1:
            return next(iter(models.values()))
        else:
            raise ValueError(f"Expected a single model, found {len(models)}")

    @model.setter
    def model(self, model: TrainableProbabilisticModelType) -> None:
        """Update the current model, using the OBJECTIVE tag."""
        if len(self.models) != 1:
            raise ValueError(f"Expected a single model, found {len(self.models)}")
        elif self.models.keys() != {OBJECTIVE}:
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
    def from_record(
        cls,
        record: Record[StateType] | FrozenRecord[StateType],
        search_space: SearchSpaceType,
        acquisition_rule: AcquisitionRule[
            TensorType | State[StateType | None, TensorType],
            SearchSpaceType,
            TrainableProbabilisticModelType,
        ]
        | None = None,
    ) -> AskTellOptimizer[SearchSpaceType, TrainableProbabilisticModelType]:
        """Creates new :class:`~AskTellOptimizer` instance from provided optimization state.
        Model training isn't triggered upon creation of the instance.

        :param record: Optimization state record.
        :param search_space: The space over which to search for the next query point.
        :param acquisition_rule: The acquisition rule, which defines how to search for a new point
            on each optimization step. Defaults to
            :class:`~trieste.acquisition.rule.EfficientGlobalOptimization` with default
            arguments.
        :return: New instance of :class:`~AskTellOptimizer`.
        """
        # we are recovering previously saved optimization state
        # so the model was already trained
        # thus there is no need to train it again

        # type ignore below is due to the fact that overloads don't allow
        # optional acquisition_rule along with acquisition_state
        return cls(
            search_space,
            record.datasets,
            cast(Mapping[Tag, TrainableProbabilisticModelType], record.models),
            acquisition_rule=acquisition_rule,  # type: ignore
            acquisition_state=record.acquisition_state,
            fit_model=False,
        )

    def to_record(self, copy: bool = True) -> Record[StateType]:
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
            state_copy = deepcopy(self._acquisition_state) if copy else self._acquisition_state
        except Exception as e:
            raise NotImplementedError(
                "Failed to copy the optimization state. Some models do not support "
                "deecopying (this is particularly common for deep neural network models). "
                "For these models, the `copy` argument of the `to_record` or `to_result` "
                "methods should be set to `False`. This means that the returned state may be "
                "modified by subsequent optimization."
            ) from e

        return Record(datasets=datasets_copy, models=models_copy, acquisition_state=state_copy)

    def to_result(self, copy: bool = True) -> OptimizationResult[StateType]:
        """Converts current state of the optimization
        into a :class:`~trieste.data.OptimizationResult` object.

        :param copy: Whether to return a copy of the current state or the original. Copying
            is not supported for all model types. However, continuing the optimization will
            modify the original state.
        :return: A :class:`~trieste.data.OptimizationResult` object.
        """
        record: Record[StateType] = self.to_record(copy=copy)
        return OptimizationResult(Ok(record), [])

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

    def tell(self, new_data: Mapping[Tag, Dataset] | Dataset) -> None:
        """Updates optimizer state with new data.

        :param new_data: New observed data.
        :raise ValueError: If keys in ``new_data`` do not match those in already built dataset.
        """
        if isinstance(new_data, Dataset):
            new_data = {OBJECTIVE: new_data}

        # The datasets must have the same keys as the existing datasets. Only exception is if
        # the existing datasets are all global, in which case the dataset will be appropriately
        # updated below for the next iteration.
        datasets_indices = {LocalizedTag.from_tag(tag).local_index for tag in self._datasets.keys()}
        if self._datasets.keys() != new_data.keys() and datasets_indices != {None}:
            raise ValueError(
                f"new_data keys {new_data.keys()} doesn't "
                f"match dataset keys {self._datasets.keys()}"
            )

        for tag, new_dataset in new_data.items():
            self._datasets[tag] += new_dataset
        self._filtered_datasets = self._acquisition_rule.filter_datasets(
            self._models, self._datasets
        )

        with Timer() as model_fitting_timer:
            for tag, model in self._models.items():
                # Always use the matching dataset to the model. If the model is
                # local, then the dataset should be too by this stage.
                dataset = self._filtered_datasets[tag]
                model.update(dataset)
                optimize_model_and_save_result(model, dataset)

        summary_writer = logging.get_tensorboard_writer()
        if summary_writer:
            with summary_writer.as_default(step=logging.get_step_number()):
                write_summary_observations(
                    self._datasets,
                    self._models,
                    new_data,
                    model_fitting_timer,
                    self._observation_plot_dfs,
                )
