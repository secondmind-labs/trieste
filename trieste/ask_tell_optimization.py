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

from typing import Dict, Generic, Mapping, Optional, TypeVar, cast

from .acquisition.rule import AcquisitionRule
from .bayesian_optimizer import Record
from .data import Dataset
from .models import ModelSpec, create_model
from .observer import OBJECTIVE
from .space import SearchSpace
from .types import TensorType
from .utils import map_values

S = TypeVar("S")
""" Unbound type variable. """

SP = TypeVar("SP", bound=SearchSpace)
""" Type variable bound to :class:`SearchSpace`. """


class AskTellOptimizer(Generic[S, SP]):
    """
    This class provides Ask/Tell optimization interface. It is designed for those use cases
    when control of the optimization loop by Trieste is impossible.
    That could happen because objective function cannot be implemented/called in Python, or if
    optimization state cannot be maintained in memory. For more details about the Bayesian Optimization routine,
    refer to :class:`BayesianOptimizer`.
    """

    def __init__(
        self,
        search_space: SP,
        datasets: Mapping[str, Dataset] | Dataset,
        model_specs: Mapping[str, ModelSpec] | ModelSpec,
        fit_model: bool,
        acquisition_rule: AcquisitionRule[S, SP],
        acquisition_state: Optional[S] = None,
    ):
        """
        :param search_space: The space over which to search.
        :param datasets: The known observer query points and observations for each tag.
        :param model_specs: The model to use for each :class:`~trieste.data.Dataset` in
            ``datasets``.
        :param fit_model: If `True`, models passed in will be optimized on the given data.
            If `False`, the models are assumed to be optimized already.
        :param acquisition_rule: The acquisition rule, which defines how to search for a new point
            on each optimization step.
        :param acquisition_state: The optional acquisition state for stateful acquisitions.
        :raise ValueError: If any of the following are true:
            - the keys in ``datasets`` and ``model_specs`` do not match
            - ``datasets`` or ``model_specs`` are empty
        """
        self._search_space = search_space
        self._acquisition_rule = acquisition_rule
        self._acquisition_state = acquisition_state

        if not datasets or not model_specs:
            raise ValueError("dicts of datasets and model_specs must be populated.")

        if isinstance(datasets, Dataset):
            datasets = {OBJECTIVE: datasets}
            # ignore below is due to MyPy not being able to handle overloads properly
            model_specs = {OBJECTIVE: model_specs}  # type: ignore

        # reassure the type checker that everything is tagged
        datasets = cast(Dict[str, Dataset], datasets)
        model_specs = cast(Dict[str, ModelSpec], model_specs)

        if datasets.keys() != model_specs.keys():
            raise ValueError(
                f"datasets and model_specs should contain the same keys. Got {datasets.keys()} and"
                f" {model_specs.keys()} respectively."
            )

        self._datasets = datasets
        self._models = map_values(create_model, model_specs)

        if fit_model:
            # there is no default value for `fit_model` parameter to force the user to think about it
            # before invoking ask/tell API
            # there is no sensible default here:
            # `False` may result in suboptimal performance
            # `True` may result in two model training calls per optimization step
            for tag, model in self._models.items():
                dataset = datasets[tag]
                model.update(dataset)
                model.optimize(dataset)

    @classmethod
    def from_record(
        cls, search_space: SP, acquisition_rule: AcquisitionRule[S, SP], record: Record[S]
    ) -> AskTellOptimizer[S, SP]:
        """Creates new :class:`~AskTellOptimizer` instance from provided optimization state.

        :param search_space: The space over which to search.
        :param acquisition_rule: The acquisition rule, which defines how to search for a new point
            on each optimization step.
        :param record: Optimization state record.
        """
        # here we are recovering previously saved optimization state
        # there the model was already trained
        # thus there is no need to train it again
        fit_model = False
        return cls(
            search_space,
            record.datasets,
            record.models,
            fit_model,
            acquisition_rule,
            record.acquisition_state,
        )

    def ask(self) -> TensorType:
        """Suggests a point (or points in batch mode) to observe by optimizing the acquisition function.
        If the acquisition is stateful, its state is saved.

        :return: A :class:`TensorType` instance representing suggested point(s).
        """
        points_or_stateful = self._acquisition_rule.acquire(
            self._search_space, self._datasets, self._models
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
                f"new_data keys {new_data.keys()} doesn't match dataset keys {self._datasets.keys()}"
            )

        self._datasets = {tag: self._datasets[tag] + new_data[tag] for tag in new_data}

        for tag, model in self._models.items():
            dataset = self._datasets[tag]
            model.update(dataset)
            model.optimize(dataset)

    def get_state(self) -> Record[S]:
        """Collects the current state of the optimization, which includes datasets, models and acquisition state (if applicable).
        :return: An optimization state record.
        """
        return Record(
            datasets=self._datasets, models=self._models, acquisition_state=self._acquisition_state
        )
