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
from abc import abstractmethod
from typing import Mapping, Sequence

import tensorflow as tf

from .function import AcquisitionFunctionBuilder, AcquisitionFunction
from ..datasets import Dataset
from ..type import QueryPoints
from ..models import ModelInterface


class Reducer(AcquisitionFunctionBuilder):
    """
    A :class:`Reducer` builds an :func:`~trieste.acquisition.AcquisitionFunction` whose output is
    calculated from the outputs of a number of other
    :func:`~trieste.acquisition.AcquisitionFunction`\ s. How these outputs are composed is
    defined by the method :meth:`_reduce`.
    """

    def __init__(self, *builders: AcquisitionFunctionBuilder):
        r"""
        :param \*builders: Acquisition function builders. At least one must be provided.
        :raise ValueError: If no builders are specified.
        """
        classname = self.__class__.__name__
        if len(builders) < 1:
            raise ValueError(f"{classname} expects at least one acquisition builder.")
        if not all([isinstance(v, AcquisitionFunctionBuilder) for v in builders]):
            raise TypeError(
                f"{classname} expects `AcquisitionFunctionBuilder` instances as inputs."
            )
        self._acquisitions = builders

    def prepare_acquisition_function(
        self, datasets: Mapping[str, Dataset], models: Mapping[str, ModelInterface]
    ) -> AcquisitionFunction:
        """
        Return an acquisition function. This acquisition function is defined by first building
        acquisition functions from each of the
        :class:`~trieste.acquisition.AcquisitionFunctionBuilder`\ s specified at
        :meth:`__init__`, then reducing, with :meth:`_reduce`, the output of each of those
        acquisition functions.

        :param datasets: The data from the observer.
        :param models: The models over each dataset in ``datasets``.
        :return: The reduced acquisition function.
        """
        functions = tuple(
            acq.prepare_acquisition_function(datasets, models) for acq in self.acquisitions
        )

        def evaluate_acquisition_function_fn(at: QueryPoints) -> tf.Tensor:
            return self._reduce_acquisition_functions(at, functions)

        return evaluate_acquisition_function_fn

    @property
    def acquisitions(self) -> Sequence[AcquisitionFunctionBuilder]:
        """ The acquisition function builders specified at class initialisation. """
        return self._acquisitions

    def _reduce_acquisition_functions(
        self, at: QueryPoints, acquisition_functions: Sequence[AcquisitionFunction]
    ) -> tf.Tensor:
        return self._reduce([fn(at) for fn in acquisition_functions])

    @abstractmethod
    def _reduce(self, inputs: Sequence[tf.Tensor]) -> tf.Tensor:
        """
        :param inputs: The output of each constituent acquisition function.
        :return: The output of the reduced acquisition function.
        """
        raise NotImplementedError()


class Sum(Reducer):
    """
    :class:`Reducer` whose resulting acquisition function returns the element-wise sum of the
    outputs of constituent acquisition functions.
    """

    def _reduce(self, inputs: Sequence[tf.Tensor]) -> tf.Tensor:
        """
        :param inputs: The outputs of each acquisition function.
        :return: The element-wise sum of the ``inputs``.
        """
        return tf.add_n(inputs)


class Product(Reducer):
    """
    :class:`Reducer` whose resulting acquisition function returns the element-wise product of the
    outputs of constituent acquisition functions.
    """

    def _reduce(self, inputs: Sequence[tf.Tensor]) -> tf.Tensor:
        """
        :param inputs: The outputs of each acquisition function.
        :return: The element-wise product of the ``inputs``.
        """
        return tf.reduce_prod(inputs, axis=0)
