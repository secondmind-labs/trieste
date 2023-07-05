# Copyright 2020 The Trieste Contributors
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
This module contains the interfaces relating to acquisition function --- functions that estimate
the utility of evaluating sets of candidate points.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Generic, Mapping, Optional

from ..data import Dataset
from ..models.interfaces import ProbabilisticModelType
from ..types import Tag, TensorType

AcquisitionFunction = Callable[[TensorType], TensorType]
"""
Type alias for acquisition functions.

An :const:`AcquisitionFunction` maps a set of `B` query points (each of dimension `D`) to a single
value that describes how useful it would be evaluate all these points together (to our goal of
optimizing the objective function). Thus, with leading dimensions, an :const:`AcquisitionFunction`
takes input shape `[..., B, D]` and returns shape `[..., 1]`.

Note that :const:`AcquisitionFunction`s which do not support batch optimization still expect inputs
with a batch dimension, i.e. an input of shape `[..., 1, D]`.
"""


class AcquisitionFunctionClass(ABC):
    """An :class:`AcquisitionFunctionClass` is an acquisition function represented using a class
    rather than as a standalone function. Using a class to represent an acquisition function
    makes it easier to update it, to avoid having to retrace the function on every call.
    """

    @abstractmethod
    def __call__(self, x: TensorType) -> TensorType:
        """Call acquisition function."""


class AcquisitionFunctionBuilder(Generic[ProbabilisticModelType], ABC):
    """An :class:`AcquisitionFunctionBuilder` builds and updates an acquisition function."""

    @abstractmethod
    def prepare_acquisition_function(
        self,
        models: Mapping[Tag, ProbabilisticModelType],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> AcquisitionFunction:
        """
        Prepare an acquisition function. We assume that this requires at least models, but
        it may sometimes also need data.

        :param models: The models for each tag.
        :param datasets: The data from the observer (optional).
        :return: An acquisition function.
        """

    def update_acquisition_function(
        self,
        function: AcquisitionFunction,
        models: Mapping[Tag, ProbabilisticModelType],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> AcquisitionFunction:
        """
        Update an acquisition function. By default this generates a new acquisition function each
        time. However, if the function is decorated with `@tf.function`, then you can override
        this method to update its variables instead and avoid retracing the acquisition function on
        every optimization loop.

        :param function: The acquisition function to update.
        :param models: The models for each tag.
        :param datasets: The data from the observer (optional).
        :return: The updated acquisition function.
        """
        return self.prepare_acquisition_function(models, datasets=datasets)


class SingleModelAcquisitionBuilder(Generic[ProbabilisticModelType], ABC):
    """
    Convenience acquisition function builder for an acquisition function (or component of a
    composite acquisition function) that requires only one model, dataset pair.
    """

    def using(self, tag: Tag) -> AcquisitionFunctionBuilder[ProbabilisticModelType]:
        """
        :param tag: The tag for the model, dataset pair to use to build this acquisition function.
        :return: An acquisition function builder that selects the model and dataset specified by
            ``tag``, as defined in :meth:`prepare_acquisition_function`.
        """

        class _Anon(AcquisitionFunctionBuilder[ProbabilisticModelType]):
            def __init__(
                self, single_builder: SingleModelAcquisitionBuilder[ProbabilisticModelType]
            ):
                self.single_builder = single_builder

            def prepare_acquisition_function(
                self,
                models: Mapping[Tag, ProbabilisticModelType],
                datasets: Optional[Mapping[Tag, Dataset]] = None,
            ) -> AcquisitionFunction:
                return self.single_builder.prepare_acquisition_function(
                    models[tag], dataset=None if datasets is None else datasets[tag]
                )

            def update_acquisition_function(
                self,
                function: AcquisitionFunction,
                models: Mapping[Tag, ProbabilisticModelType],
                datasets: Optional[Mapping[Tag, Dataset]] = None,
            ) -> AcquisitionFunction:
                return self.single_builder.update_acquisition_function(
                    function, models[tag], dataset=None if datasets is None else datasets[tag]
                )

            def __repr__(self) -> str:
                return f"{self.single_builder!r} using tag {tag!r}"

        return _Anon(self)

    @abstractmethod
    def prepare_acquisition_function(
        self,
        model: ProbabilisticModelType,
        dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        """
        :param model: The model.
        :param dataset: The data to use to build the acquisition function (optional).
        :return: An acquisition function.
        """

    def update_acquisition_function(
        self,
        function: AcquisitionFunction,
        model: ProbabilisticModelType,
        dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        """
        :param function: The acquisition function to update.
        :param model: The model.
        :param dataset: The data from the observer (optional).
        :return: The updated acquisition function.
        """
        return self.prepare_acquisition_function(model, dataset=dataset)


class GreedyAcquisitionFunctionBuilder(Generic[ProbabilisticModelType], ABC):
    """
    A :class:`GreedyAcquisitionFunctionBuilder` builds an acquisition function
    suitable for greedily building batches for batch Bayesian
    Optimization. A :class:`GreedyAcquisitionFunctionBuilder` differs
    from an :class:`AcquisitionFunctionBuilder` by requiring that a set
    of pending points is passed to the builder. Note that this acquisition function
    is typically called `B` times each Bayesian optimization step, when building batches
    of size `B`.
    """

    @abstractmethod
    def prepare_acquisition_function(
        self,
        models: Mapping[Tag, ProbabilisticModelType],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
        pending_points: Optional[TensorType] = None,
    ) -> AcquisitionFunction:
        """
        Generate a new acquisition function. The first time this is called, ``pending_points``
        will be `None`. Subsequent calls will be via ``update_acquisition_function`` below,
        unless that has been overridden.

        :param models: The models over each tag.
        :param datasets: The data from the observer (optional).
        :param pending_points: Points already chosen to be in the current batch (of shape [M,D]),
            where M is the number of pending points and D is the search space dimension.
        :return: An acquisition function.
        """

    def update_acquisition_function(
        self,
        function: AcquisitionFunction,
        models: Mapping[Tag, ProbabilisticModelType],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
        pending_points: Optional[TensorType] = None,
        new_optimization_step: bool = True,
    ) -> AcquisitionFunction:
        """
        Update an acquisition function. By default this generates a new acquisition function each
        time. However, if the function is decorated with`@tf.function`, then you can override
        this method to update its variables instead and avoid retracing the acquisition function on
        every optimization loop.

        :param function: The acquisition function to update.
        :param models: The models over each tag.
        :param datasets: The data from the observer (optional).
        :param pending_points: Points already chosen to be in the current batch (of shape [M,D]),
            where M is the number of pending points and D is the search space dimension.
        :param new_optimization_step: Indicates whether this call to update_acquisition_function
            is to start of a new optimization step, of to continue collecting batch of points
            for the current step. Defaults to ``True``.
        :return: The updated acquisition function.
        """
        return self.prepare_acquisition_function(
            models, datasets=datasets, pending_points=pending_points
        )


class SingleModelGreedyAcquisitionBuilder(Generic[ProbabilisticModelType], ABC):
    """
    Convenience acquisition function builder for a greedy acquisition function (or component of a
    composite greedy acquisition function) that requires only one model, dataset pair.
    """

    def using(self, tag: Tag) -> GreedyAcquisitionFunctionBuilder[ProbabilisticModelType]:
        """
        :param tag: The tag for the model, dataset pair to use to build this acquisition function.
        :return: An acquisition function builder that selects the model and dataset specified by
            ``tag``, as defined in :meth:`prepare_acquisition_function`.
        """

        class _Anon(GreedyAcquisitionFunctionBuilder[ProbabilisticModelType]):
            def __init__(
                self, single_builder: SingleModelGreedyAcquisitionBuilder[ProbabilisticModelType]
            ):
                self.single_builder = single_builder

            def prepare_acquisition_function(
                self,
                models: Mapping[Tag, ProbabilisticModelType],
                datasets: Optional[Mapping[Tag, Dataset]] = None,
                pending_points: Optional[TensorType] = None,
            ) -> AcquisitionFunction:
                return self.single_builder.prepare_acquisition_function(
                    models[tag],
                    dataset=None if datasets is None else datasets[tag],
                    pending_points=pending_points,
                )

            def update_acquisition_function(
                self,
                function: AcquisitionFunction,
                models: Mapping[Tag, ProbabilisticModelType],
                datasets: Optional[Mapping[Tag, Dataset]] = None,
                pending_points: Optional[TensorType] = None,
                new_optimization_step: bool = True,
            ) -> AcquisitionFunction:
                return self.single_builder.update_acquisition_function(
                    function,
                    models[tag],
                    dataset=None if datasets is None else datasets[tag],
                    pending_points=pending_points,
                    new_optimization_step=new_optimization_step,
                )

            def __repr__(self) -> str:
                return f"{self.single_builder!r} using tag {tag!r}"

        return _Anon(self)

    @abstractmethod
    def prepare_acquisition_function(
        self,
        model: ProbabilisticModelType,
        dataset: Optional[Dataset] = None,
        pending_points: Optional[TensorType] = None,
    ) -> AcquisitionFunction:
        """
        :param model: The model.
        :param dataset: The data from the observer (optional).
        :param pending_points: Points already chosen to be in the current batch (of shape [M,D]),
            where M is the number of pending points and D is the search space dimension.
        :return: An acquisition function.
        """

    def update_acquisition_function(
        self,
        function: AcquisitionFunction,
        model: ProbabilisticModelType,
        dataset: Optional[Dataset] = None,
        pending_points: Optional[TensorType] = None,
        new_optimization_step: bool = True,
    ) -> AcquisitionFunction:
        """
        :param function: The acquisition function to update.
        :param model: The model.
        :param dataset: The data from the observer (optional).
        :param pending_points: Points already chosen to be in the current batch (of shape [M,D]),
            where M is the number of pending points and D is the search space dimension.
        :param new_optimization_step: Indicates whether this call to update_acquisition_function
            is to start of a new optimization step, of to continue collecting batch of points
            for the current step. Defaults to ``True``.
        :return: The updated acquisition function.
        """
        return self.prepare_acquisition_function(
            model,
            dataset=dataset,
            pending_points=pending_points,
        )


class VectorizedAcquisitionFunctionBuilder(AcquisitionFunctionBuilder[ProbabilisticModelType]):
    """
    An :class:`VectorizedAcquisitionFunctionBuilder` builds and updates a vectorized
    acquisition function These differ from normal acquisition functions only by their output shape:
    rather than returning a single value, they return one value per potential query point.
    Thus, with leading dimensions, they take input shape `[..., B, D]` and returns shape `[..., B]`.
    """


class SingleModelVectorizedAcquisitionBuilder(
    SingleModelAcquisitionBuilder[ProbabilisticModelType]
):
    """
    Convenience acquisition function builder for vectorized acquisition functions (or component
    of a composite vectorized acquisition function) that requires only one model, dataset pair.
    """

    def using(self, tag: Tag) -> AcquisitionFunctionBuilder[ProbabilisticModelType]:
        """
        :param tag: The tag for the model, dataset pair to use to build this acquisition function.
        :return: An acquisition function builder that selects the model and dataset specified by
            ``tag``, as defined in :meth:`prepare_acquisition_function`.
        """

        class _Anon(VectorizedAcquisitionFunctionBuilder[ProbabilisticModelType]):
            def __init__(
                self,
                single_builder: SingleModelVectorizedAcquisitionBuilder[ProbabilisticModelType],
            ):
                self.single_builder = single_builder

            def prepare_acquisition_function(
                self,
                models: Mapping[Tag, ProbabilisticModelType],
                datasets: Optional[Mapping[Tag, Dataset]] = None,
            ) -> AcquisitionFunction:
                return self.single_builder.prepare_acquisition_function(
                    models[tag], dataset=None if datasets is None else datasets[tag]
                )

            def update_acquisition_function(
                self,
                function: AcquisitionFunction,
                models: Mapping[Tag, ProbabilisticModelType],
                datasets: Optional[Mapping[Tag, Dataset]] = None,
            ) -> AcquisitionFunction:
                return self.single_builder.update_acquisition_function(
                    function, models[tag], dataset=None if datasets is None else datasets[tag]
                )

            def __repr__(self) -> str:
                return f"{self.single_builder!r} using tag {tag!r}"

        return _Anon(self)


PenalizationFunction = Callable[[TensorType], TensorType]
"""
An :const:`PenalizationFunction` maps a query point (of dimension `D`) to a single
value that described how heavily it should be penalized (a positive quantity).
As penalization is applied multiplicatively to acquisition functions, small
penalization outputs correspond to a stronger penalization effect. Thus, with
leading dimensions, an :const:`PenalizationFunction` takes input
shape `[..., 1, D]` and returns shape `[..., 1]`.
"""


class UpdatablePenalizationFunction(ABC):
    """An :class:`UpdatablePenalizationFunction` builds and updates a penalization function.
    Defining a penalization function that can be updated avoids having to retrace on every call."""

    @abstractmethod
    def __call__(self, x: TensorType) -> TensorType:
        """Call penalization function.."""

    @abstractmethod
    def update(
        self,
        pending_points: TensorType,
        lipschitz_constant: TensorType,
        eta: TensorType,
    ) -> None:
        """Update penalization function."""
