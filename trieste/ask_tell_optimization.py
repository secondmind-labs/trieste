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

import numpy as np
import tensorflow as tf

from . import logging
from .acquisition.rule import AcquisitionRule, EfficientGlobalOptimization
from .bayesian_optimizer import FrozenRecord, OptimizationResult, Record
from .data import Dataset
from .models import ModelSpec, TrainableProbabilisticModel, create_model
from .models.config import ModelConfigType
from .observer import OBJECTIVE
from .space import SearchSpace
from .types import State, TensorType
from .utils import Ok, Timer, map_values

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

    @property
    def datasets(self) -> Mapping[str, Dataset]:
        """The current datasets."""
        return self._datasets

    @property
    def dataset(self) -> Dataset:
        """The current dataset when there is just one dataset."""
        if len(self.datasets) == 1:
            return next(iter(self.datasets.values()))
        else:
            raise ValueError(f"Expected a single dataset, found {len(self.datasets)}")

    @property
    def models(self) -> Mapping[str, TrainableProbabilisticModel]:
        """The current models."""
        return self._models

    @property
    def model(self) -> TrainableProbabilisticModel:
        """The current model when there is just one model."""
        if len(self.models) == 1:
            return next(iter(self.models.values()))
        else:
            raise ValueError(f"Expected a single model, found {len(self.models)}")

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

        with Timer() as query_point_generation_timer:
            points_or_stateful = self._acquisition_rule.acquire(
                self._search_space, self._models, datasets=self._datasets
            )

        if callable(points_or_stateful):
            self._acquisition_state, query_points = points_or_stateful(self._acquisition_state)
        else:
            query_points = points_or_stateful

        summary_writer = logging.get_tensorboard_writer()
        if summary_writer:
            with summary_writer.as_default(step=logging.get_step_number()):
                if tf.rank(query_points) == 2:
                    for i in tf.range(tf.shape(query_points)[1]):
                        if len(query_points) == 1:
                            logging.scalar(f"query_points/[{i}]", float(query_points[0, i]))
                        else:
                            logging.histogram(f"query_points/[{i}]", query_points[:, i])
                logging.scalar(
                    "wallclock/query_point_generation",
                    query_point_generation_timer.time,
                )

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

        for tag in self._datasets:
            self._datasets[tag] += new_data[tag]

        with Timer() as model_fitting_timer:
            for tag, model in self._models.items():
                dataset = self._datasets[tag]
                model.update(dataset)
                model.optimize(dataset)

        summary_writer = logging.get_tensorboard_writer()
        if summary_writer:
            with summary_writer.as_default(step=logging.get_step_number()):
                for tag in self._datasets:
                    with tf.name_scope(f"{tag}.model"):
                        self._models[tag].log()
                    output_dim = tf.shape(new_data[tag].observations)[-1]
                    for i in tf.range(output_dim):
                        suffix = f"[{i}]" if output_dim > 1 else ""
                        if tf.size(new_data[tag].observations) > 0:
                            logging.histogram(
                                f"{tag}.observation{suffix}/new_observations",
                                new_data[tag].observations[..., i],
                            )
                            logging.scalar(
                                f"{tag}.observation{suffix}/best_new_observation",
                                np.min(new_data[tag].observations[..., i]),
                            )
                        if tf.size(self._datasets[tag].observations) > 0:
                            logging.scalar(
                                f"{tag}.observation{suffix}/best_overall",
                                np.min(self._datasets[tag].observations[..., i]),
                            )
                    logging.scalar("wallclock/model_fitting", model_fitting_timer.time)
