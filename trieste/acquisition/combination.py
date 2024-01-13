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
from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping, Sequence
from typing import Callable, Optional

import tensorflow as tf

from ..data import Dataset
from ..models import ProbabilisticModelType
from ..types import Tag, TensorType
from .interface import AcquisitionFunction, AcquisitionFunctionBuilder


class Reducer(AcquisitionFunctionBuilder[ProbabilisticModelType]):
    r"""
    A :class:`Reducer` builds an :const:`~trieste.acquisition.AcquisitionFunction` whose output is
    calculated from the outputs of a number of other
    :const:`~trieste.acquisition.AcquisitionFunction`\ s. How these outputs are composed is defined
    by the method :meth:`_reduce`.
    """

    def __init__(self, *builders: AcquisitionFunctionBuilder[ProbabilisticModelType]):
        r"""
        :param \*builders: Acquisition function builders. At least one must be provided.
        :raise `~tf.errors.InvalidArgumentError`: If no builders are specified.
        """
        tf.debugging.assert_positive(
            len(builders), "At least one acquisition builder expected, got none."
        )

        self._acquisitions = builders

    def __repr__(self) -> str:
        """"""
        builders = ", ".join(map(repr, self._acquisitions))
        return f"{self.__class__.__name__}({builders})"

    def prepare_acquisition_function(
        self,
        models: Mapping[Tag, ProbabilisticModelType],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> AcquisitionFunction:
        r"""
        Return an acquisition function. This acquisition function is defined by first building
        acquisition functions from each of the
        :class:`~trieste.acquisition.AcquisitionFunctionBuilder`\ s specified at
        :meth:`__init__`, then reducing, with :meth:`_reduce`, the output of each of those
        acquisition functions.

        :param datasets: The data from the observer.
        :param models: The models over each dataset in ``datasets``.
        :return: The reduced acquisition function.
        """
        self.functions = tuple(
            acq.prepare_acquisition_function(models, datasets=datasets) for acq in self.acquisitions
        )

        def evaluate_acquisition_function_fn(at: TensorType) -> TensorType:
            return self._reduce_acquisition_functions(at, self.functions)

        return evaluate_acquisition_function_fn

    def update_acquisition_function(
        self,
        function: AcquisitionFunction,
        models: Mapping[Tag, ProbabilisticModelType],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
    ) -> AcquisitionFunction:
        """
        :param function: The acquisition function to update.
        :param models: The model.
        :param datasets: Unused.
        """
        self.functions = tuple(
            acq.update_acquisition_function(function, models, datasets=datasets)
            for function, acq in zip(self.functions, self.acquisitions)
        )

        def evaluate_acquisition_function_fn(at: TensorType) -> TensorType:
            return self._reduce_acquisition_functions(at, self.functions)

        return evaluate_acquisition_function_fn

    @property
    def acquisitions(self) -> Sequence[AcquisitionFunctionBuilder[ProbabilisticModelType]]:
        """The acquisition function builders specified at class initialisation."""
        return self._acquisitions

    def _reduce_acquisition_functions(
        self, at: TensorType, acquisition_functions: Sequence[AcquisitionFunction]
    ) -> TensorType:
        return self._reduce([fn(at) for fn in acquisition_functions])

    @abstractmethod
    def _reduce(self, inputs: Sequence[TensorType]) -> TensorType:
        """
        :param inputs: The output of each constituent acquisition function.
        :return: The output of the reduced acquisition function.
        """
        raise NotImplementedError()


class Sum(Reducer[ProbabilisticModelType]):
    """
    :class:`Reducer` whose resulting acquisition function returns the element-wise sum of the
    outputs of constituent acquisition functions.
    """

    def _reduce(self, inputs: Sequence[TensorType]) -> TensorType:
        """
        :param inputs: The outputs of each acquisition function.
        :return: The element-wise sum of the ``inputs``.
        """
        return tf.add_n(inputs)


class Product(Reducer[ProbabilisticModelType]):
    """
    :class:`Reducer` whose resulting acquisition function returns the element-wise product of the
    outputs of constituent acquisition functions.
    """

    def _reduce(self, inputs: Sequence[TensorType]) -> TensorType:
        """
        :param inputs: The outputs of each acquisition function.
        :return: The element-wise product of the ``inputs``.
        """
        return tf.reduce_prod(inputs, axis=0)


class Map(Reducer[ProbabilisticModelType]):
    """
    :class:`Reducer` that accepts just one acquisition function builder and applies a
    given function to its output. For example ``Map(lambda x: -x, builder)`` would generate
    an acquisition function that returns the negative of the output of ``builder``.
    """

    def __init__(
        self,
        map_fn: Callable[[TensorType], TensorType],
        builder: AcquisitionFunctionBuilder[ProbabilisticModelType],
    ):
        """
        :param map_fn: Function to apply.
        :param builder: Acquisition function builder.
        """
        super().__init__(builder)
        self._map_fn = map_fn

    def _reduce(self, inputs: Sequence[TensorType]) -> TensorType:
        """
        :param inputs: The outputs of the acquisition function.
        :return: The result of applying the map function to ``inputs``.
        """
        tf.debugging.assert_equal(len(inputs), 1)
        return self._map_fn(inputs[0])
