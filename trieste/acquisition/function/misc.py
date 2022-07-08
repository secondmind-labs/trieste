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
This module contains miscellaneous acquisition function builders, typically helpers for other
acquisition functions.
"""

from __future__ import annotations

from typing import Optional

import tensorflow as tf
import tensorflow_probability as tfp

from ...data import Dataset
from ...models import ProbabilisticModel
from ...types import TensorType
from ..interface import AcquisitionFunction, ProbabilisticModelType, SingleModelAcquisitionBuilder


class ProbabilityOfFeasibility(SingleModelAcquisitionBuilder[ProbabilisticModel]):
    r"""
    Builder for the :func:`probability_of_feasibility` acquisition function, defined in
    :cite:`gardner14` as

    .. math::

        \int_{-\infty}^{\tau} p(c(\mathbf{x}) | \mathbf{x}, \mathcal{D}) \mathrm{d} c(\mathbf{x})
        \qquad ,

    where :math:`\tau` is a threshold. Values below the threshold are considered feasible by the
    constraint function. See also :cite:`schonlau1998global` for details.
    """

    def __init__(self, threshold: float | TensorType):
        """
        :param threshold: The (scalar) probability of feasibility threshold.
        :raise ValueError (or InvalidArgumentError): If ``threshold`` is not a scalar.
        """
        tf.debugging.assert_scalar(threshold)

        self._threshold = threshold

    def __repr__(self) -> str:
        """"""
        return f"ProbabilityOfFeasibility({self._threshold!r})"

    @property
    def threshold(self) -> float | TensorType:
        """The probability of feasibility threshold."""
        return self._threshold

    def prepare_acquisition_function(
        self,
        model: ProbabilisticModel,
        dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        """
        :param model: The model.
        :param dataset: Unused.
        :return: The probability of feasibility function. This function will raise
            :exc:`ValueError` or :exc:`~tf.errors.InvalidArgumentError` if used with a batch size
            greater than one.
        """
        return probability_of_feasibility(model, self.threshold)

    def update_acquisition_function(
        self,
        function: AcquisitionFunction,
        model: ProbabilisticModel,
        dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        """
        :param function: The acquisition function to update.
        :param model: The model.
        :param dataset: Unused.
        """
        return function  # no need to update anything


def probability_of_feasibility(
    model: ProbabilisticModel, threshold: float | TensorType
) -> AcquisitionFunction:
    r"""
    The probability of feasibility acquisition function defined in :cite:`gardner14` as

    .. math::

        \int_{-\infty}^{\tau} p(c(\mathbf{x}) | \mathbf{x}, \mathcal{D}) \mathrm{d} c(\mathbf{x})
        \qquad ,

    where :math:`\tau` is a threshold. Values below the threshold are considered feasible by the
    constraint function.

    :param model: The model of the objective function.
    :param threshold: The (scalar) probability of feasibility threshold.
    :return: The probability of feasibility function. This function will raise
        :exc:`ValueError` or :exc:`~tf.errors.InvalidArgumentError` if used with a batch size
        greater than one.
    :raise ValueError or tf.errors.InvalidArgumentError: If ``threshold`` is not a scalar.
    """
    tf.debugging.assert_scalar(threshold)

    @tf.function
    def acquisition(x: TensorType) -> TensorType:
        tf.debugging.assert_shapes(
            [(x, [..., 1, None])],
            message="This acquisition function only supports batch sizes of one.",
        )
        mean, var = model.predict(tf.squeeze(x, -2))
        distr = tfp.distributions.Normal(mean, tf.sqrt(var))
        return distr.cdf(tf.cast(threshold, x.dtype))

    return acquisition


class MakePositive(SingleModelAcquisitionBuilder[ProbabilisticModelType]):
    r"""
    Converts an acquisition function builder into one that only returns positive values, via
    :math:`x \mapsto \log(1 + \exp(x))`.

    This is sometimes a useful transformation: for example, converting non-batch acquisition
    functions into batch acquisition functions with local penalization requires functions
    that only return positive values.
    """

    def __init__(
        self,
        base_acquisition_function_builder: SingleModelAcquisitionBuilder[ProbabilisticModelType],
    ) -> None:
        """
        :param base_acquisition_function_builder: Base acquisition function to be made positive.
        """
        self._base_builder = base_acquisition_function_builder

    def __repr__(self) -> str:
        """"""
        return f"MakePositive({self._base_builder})"

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
        self._base_function = self._base_builder.prepare_acquisition_function(model, dataset)

        @tf.function
        def acquisition(x: TensorType) -> TensorType:
            return tf.math.log(1 + tf.math.exp(self._base_function(x)))

        return acquisition

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
        up_fn = self._base_builder.update_acquisition_function(self._base_function, model, dataset)
        if up_fn is self._base_function:
            return function
        else:
            self._base_function = up_fn

            @tf.function
            def acquisition(x: TensorType) -> TensorType:
                return tf.math.log(1 + tf.math.exp(self._base_function(x)))

            return acquisition
