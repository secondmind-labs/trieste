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

import copy
from typing import Dict, Generic, Mapping, TypeVar, cast, overload

import numpy as np
import tensorflow as tf

from .acquisition.rule import AcquisitionRule, EfficientGlobalOptimization
from .bayesian_optimizer import OptimizationResult, Record
from .data import Dataset
from .logging import get_step_number, get_tensorboard_writer
from .models import ModelSpec, TrainableProbabilisticModel, create_model
from .models.config import ModelConfigType
from .observer import OBJECTIVE
from .space import SearchSpace
from .types import State, TensorType
from .utils import Ok, map_values

StateType = TypeVar("StateType")
""" Unbound type variable. """

SearchSpaceType = TypeVar("SearchSpaceType", bound=SearchSpace)
""" Type variable bound to :class:`SearchSpace`. """

TrainableProbabilisticModelType = TypeVar(
    "TrainableProbabilisticModelType", bound=TrainableProbabilisticModel, contravariant=True
)
""" Contravariant type variable bound to :class:`TrainableProbabilisticModel`. """


class AskTellOptimizer(Generic[SearchSpaceType]):
    """
    This class provides Ask/Tell optimization interface. It is designed for those use cases
    when control of the optimization loop by Trieste is impossible or not desirable.
    For more details about the Bayesian Optimization routine, refer to :class:`BayesianOptimizer`.
    """

    @overload
    def __init__(
        self,
        search_space: SearchSpaceType,
        datasets: Mapping[str, Dataset],
        model_specs: Mapping[str, TrainableProbabilisticModelType],
        *,
        fit_model: bool = True,
    ):
        ...

    @overload
    def __init__(
        self,
        search_space: SearchSpaceType,
        datasets: Mapping[str, Dataset],
        model_specs: Mapping[str, ModelConfigType],
        *,
        fit_model: bool = True,
    ):
        ...

    @overload
    def __init__(
        self,
        search_space: SearchSpaceType,
        datasets: Mapping[str, Dataset],
        model_specs: Mapping[str, TrainableProbabilisticModelType],
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
        datasets: Mapping[str, Dataset],
        model_specs: Mapping[str, ModelConfigType],
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
        datasets: Mapping[str, Dataset],
        model_specs: Mapping[str, TrainableProbabilisticModelType],
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
        datasets: Mapping[str, Dataset],
        model_specs: Mapping[str, ModelConfigType],
        acquisition_rule: AcquisitionRule[
            State[StateType | None, TensorType], SearchSpaceType, TrainableProbabilisticModelType
        ],
        acquisition_state: StateType | None = None,
        *,
        fit_model: bool = True,
    ):
        ...

    @overload
    def __init__(
        self,
        search_space: SearchSpaceType,
        datasets: Dataset,
        model_specs: TrainableProbabilisticModelType,
        *,
        fit_model: bool = True,
    ):
        ...

    @overload
    def __init__(
        self,
        search_space: SearchSpaceType,
        datasets: Dataset,
        model_specs: ModelConfigType,
        *,
        fit_model: bool = True,
    ):
        ...

    @overload
    def __init__(
        self,
        search_space: SearchSpaceType,
        datasets: Dataset,
        model_specs: TrainableProbabilisticModelType,
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
        model_specs: TrainableProbabilisticModelType,
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
        model_specs: TrainableProbabilisticModelType,
        acquisition_rule: AcquisitionRule[
            State[StateType | None, TensorType], SearchSpaceType, TrainableProbabilisticModelType
        ],
        acquisition_state: StateType | None = None,
        *,
        fit_model: bool = True,
    ):
        ...

    @overload
    def __init__(
        self,
        search_space: SearchSpaceType,
        datasets: Dataset,
        model_specs: ModelConfigType,
        acquisition_rule: AcquisitionRule[
            State[StateType | None, TensorType], SearchSpaceType, TrainableProbabilisticModelType
        ],
        acquisition_state: StateType | None,
        *,
        fit_model: bool = True,
    ):
        ...

    def __init__(
        self,
        search_space: SearchSpaceType,
        datasets: Mapping[str, Dataset] | Dataset,
        model_specs: Mapping[str, ModelSpec] | ModelSpec,
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
        :param model_specs: The model to use for each :class:`~trieste.data.Dataset` in
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
            - the keys in ``datasets`` and ``model_specs`` do not match
            - ``datasets`` or ``model_specs`` are empty
            - default acquisition is used but incompatible with other inputs
        """
        self._search_space = search_space
        self._acquisition_state = acquisition_state

        if not datasets or not model_specs:
            raise ValueError("dicts of datasets and model_specs must be populated.")

        if isinstance(datasets, Dataset):
            datasets = {OBJECTIVE: datasets}
            model_specs = {OBJECTIVE: model_specs}

        # reassure the type checker that everything is tagged
        datasets = cast(Dict[str, Dataset], datasets)
        model_specs = cast(Dict[str, ModelSpec], model_specs)

        if datasets.keys() != model_specs.keys():
            raise ValueError(
                f"datasets and model_specs should contain the same keys. Got {datasets.keys()} and"
                f" {model_specs.keys()} respectively."
            )

        self._datasets = datasets
        self._models = cast(
            Dict[str, TrainableProbabilisticModelType], map_values(create_model, model_specs)
        )

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

        if fit_model:
            for tag, model in self._models.items():
                dataset = datasets[tag]
                model.update(dataset)
                model.optimize(dataset)

    def __repr__(self) -> str:
        """Print-friendly string representation"""
        return f"""AskTellOptimizer({self._search_space!r}, {self._datasets!r},
               {self._models!r}, {self._acquisition_rule!r}), "
               {self._acquisition_state!r}"""

    @classmethod
    def from_record(
        cls,
        record: Record[StateType],
        search_space: SearchSpaceType,
        acquisition_rule: AcquisitionRule[
            TensorType | State[StateType | None, TensorType],
            SearchSpaceType,
            TrainableProbabilisticModelType,
        ]
        | None = None,
    ) -> AskTellOptimizer[SearchSpaceType]:
        """Creates new :class:`~AskTellOptimizer` instance from provided optimization state.
        Model training isn't triggered upon creation of the instance.

        :param record: Optimization state record.
        :param search_space: The space over which to search for the next query point.
        :param acquisition_rule: The acquisition rule, which defines how to search for a new point
            on each optimization step. Defaults to
            :class:`~trieste.acquisition.rule.EfficientGlobalOptimization` with default
            arguments.
        """
        # we are recovering previously saved optimization state
        # so the model was already trained
        # thus there is no need to train it again

        # type ignore below is due to the fact that overloads don't allow
        # optional acquisition_rule along with acquisition_state
        return cls(
            search_space,
            record.datasets,
            cast(Mapping[str, TrainableProbabilisticModelType], record.models),
            acquisition_rule=acquisition_rule,  # type: ignore
            acquisition_state=record.acquisition_state,
            fit_model=False,
        )

    def to_record(self) -> Record[StateType]:
        """Collects the current state of the optimization, which includes datasets,
        models and acquisition state (if applicable).

        :return: An optimization state record.
        """
        models_copy = copy.deepcopy(self._models)
        acquisition_state_copy = copy.deepcopy(self._acquisition_state)
        return Record(
            datasets=self._datasets, models=models_copy, acquisition_state=acquisition_state_copy
        )

    def to_result(self) -> OptimizationResult[StateType]:
        """Converts current state of the optimization
        into a :class:`~trieste.data.OptimizationResult` object."""
        record: Record[StateType] = self.to_record()
        return OptimizationResult(Ok(record), [])

    def ask(self) -> TensorType:
        """Suggests a point (or points in batch mode) to observe by optimizing the acquisition function.
        If the acquisition is stateful, its state is saved.

        :return: A :class:`TensorType` instance representing suggested point(s).
        """
        # This trick deserves a comment to explain what's going on
        # acquisition_rule.acquire can return different things:
        # - when acquisition has no state attached, it returns just points
        # - when acquisition has state, it returns a Callable
        #   which, when called, returns state and points
        # so code below is needed to cater for both cases

        points_or_stateful = self._acquisition_rule.acquire(
            self._search_space, self._models, datasets=self._datasets
        )

        if callable(points_or_stateful):
            self._acquisition_state, query_points = points_or_stateful(self._acquisition_state)
        else:
            query_points = points_or_stateful

        return query_points

    def tell(self, new_data: Mapping[str, Dataset] | Dataset) -> None:
        """Updates optimizer state with new data.

        :param new_data: New observed data.
        :raise ValueError: If keys in ``new_data`` do not match those in already built dataset.
        """
        if isinstance(new_data, Dataset):
            new_data = {OBJECTIVE: new_data}

        if self._datasets.keys() != new_data.keys():
            raise ValueError(
                f"new_data keys {new_data.keys()} doesn't "
                f"match dataset keys {self._datasets.keys()}"
            )

        self._datasets = {tag: self._datasets[tag] + new_data[tag] for tag in new_data}

        for tag, model in self._models.items():
            dataset = self._datasets[tag]
            model.update(dataset)
            model.optimize(dataset)

        summary_writer = get_tensorboard_writer()
        step_number = get_step_number()
        if summary_writer:
            with summary_writer.as_default():
                for tag in self._datasets:
                    with tf.name_scope(f"{tag}.model"):
                        self._models[tag].log()
                    tf.summary.scalar(
                        f"{tag}.observation.best_overall",
                        np.min(self._datasets[tag].observations),
                        step=step_number,
                    )
                    tf.summary.scalar(
                        f"{tag}.observation.best_new",
                        np.min(new_data[tag].observations),
                        step=step_number,
                    )
