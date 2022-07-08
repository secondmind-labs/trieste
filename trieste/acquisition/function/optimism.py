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
This module contains optimism based acquisition function builders, which build and define our
acquisition functions --- functions that estimate the utility of evaluating sets of candidate
points.
"""

from __future__ import annotations

from typing import Optional

import tensorflow as tf
import tensorflow_probability as tfp

from ...data import Dataset
from ...models import ProbabilisticModel
from ...space import SearchSpace
from ...types import TensorType
from ..interface import (
    AcquisitionFunction,
    AcquisitionFunctionClass,
    SingleModelAcquisitionBuilder,
    SingleModelVectorizedAcquisitionBuilder,
)


class NegativeLowerConfidenceBound(SingleModelAcquisitionBuilder[ProbabilisticModel]):
    """
    Builder for the negative of the lower confidence bound. The lower confidence bound is typically
    minimised, so the negative is suitable for maximisation.
    """

    def __init__(self, beta: float = 1.96):
        """
        :param beta: Weighting given to the variance contribution to the lower confidence bound.
            Must not be negative.
        """
        self._beta = beta

    def __repr__(self) -> str:
        """"""
        return f"NegativeLowerConfidenceBound({self._beta!r})"

    def prepare_acquisition_function(
        self,
        model: ProbabilisticModel,
        dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        """
        :param model: The model.
        :param dataset: Unused.
        :return: The negative lower confidence bound function. This function will raise
            :exc:`ValueError` or :exc:`~tf.errors.InvalidArgumentError` if used with a batch size
            greater than one.
        :raise ValueError: If ``beta`` is negative.
        """
        lcb = lower_confidence_bound(model, self._beta)
        return tf.function(lambda at: -lcb(at))

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


class NegativePredictiveMean(NegativeLowerConfidenceBound):
    """
    Builder for the negative of the predictive mean. The predictive mean is minimised on minimising
    the objective function. The negative predictive mean is therefore maximised.
    """

    def __init__(self) -> None:
        super().__init__(beta=0.0)

    def __repr__(self) -> str:
        """"""
        return "NegativePredictiveMean()"


def lower_confidence_bound(model: ProbabilisticModel, beta: float) -> AcquisitionFunction:
    r"""
    The lower confidence bound (LCB) acquisition function for single-objective global optimization.

    .. math:: x^* \mapsto \mathbb{E} [f(x^*)|x, y] - \beta \sqrt{ \mathrm{Var}[f(x^*)|x, y] }

    See :cite:`Srinivas:2010` for details.

    :param model: The model of the objective function.
    :param beta: The weight to give to the standard deviation contribution of the LCB. Must not be
        negative.
    :return: The lower confidence bound function. This function will raise
        :exc:`ValueError` or :exc:`~tf.errors.InvalidArgumentError` if used with a batch size
        greater than one.
    :raise tf.errors.InvalidArgumentError: If ``beta`` is negative.
    """
    tf.debugging.assert_non_negative(
        beta, message="Standard deviation scaling parameter beta must not be negative"
    )

    @tf.function
    def acquisition(x: TensorType) -> TensorType:
        tf.debugging.assert_shapes(
            [(x, [..., 1, None])],
            message="This acquisition function only supports batch sizes of one.",
        )
        mean, variance = model.predict(tf.squeeze(x, -2))
        return mean - beta * tf.sqrt(variance)

    return acquisition


class MultipleOptimismNegativeLowerConfidenceBound(
    SingleModelVectorizedAcquisitionBuilder[ProbabilisticModel]
):
    """
    A simple parallelization of the lower confidence bound acquisition function that produces
    a vectorized acquisition function which can efficiently optimized even for large batches.

    See :cite:`torossian2020bayesian` for details.
    """

    def __init__(self, search_space: SearchSpace):
        """
        :param search_space: The global search space over which the optimisation is defined.
        """
        self._search_space = search_space

    def __repr__(self) -> str:
        """"""
        return f"MultipleOptimismNegativeLowerConfidenceBound({self._search_space!r})"

    def prepare_acquisition_function(
        self,
        model: ProbabilisticModel,
        dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        """
        :param model: The model.
        :param dataset: Unused.
        :return: The multiple optimism negative lower confidence bound function.
        """
        return multiple_optimism_lower_confidence_bound(model, self._search_space.dimension)

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
        tf.debugging.Assert(isinstance(function, multiple_optimism_lower_confidence_bound), [])
        return function  # nothing to update


class multiple_optimism_lower_confidence_bound(AcquisitionFunctionClass):
    r"""
    The multiple optimism lower confidence bound (MOLCB) acquisition function for single-objective
    global optimization.

    Each batch dimension of this acquisiton function correponds to a lower confidence bound
    acquisition function with different beta values, i.e. each point in a batch chosen by this
    acquisition function lies on a gradient of exploration/exploitation trade-offs.

    We choose the different beta values following the cdf method of :cite:`torossian2020bayesian`.
    See their paper for more details.
    """

    def __init__(self, model: ProbabilisticModel, search_space_dim: int):
        """
        :param model: The model of the objective function.
        :param search_space_dim: The dimensions of the optimisation problem's search space.
        :raise tf.errors.InvalidArgumentError: If ``search_space_dim`` is not postive.
        """

        tf.debugging.assert_positive(search_space_dim)
        self._search_space_dim = search_space_dim

        self._model = model
        self._initialized = tf.Variable(False)  # Keep track of when we need to resample
        self._betas = tf.Variable(tf.ones([0], dtype=tf.float64), shape=[None])  # [0] lazy init

    @tf.function
    def __call__(self, x: TensorType) -> TensorType:

        batch_size = tf.shape(x)[-2]
        tf.debugging.assert_positive(batch_size)

        if self._initialized:  # check batch size hasnt changed during BO
            tf.debugging.assert_equal(
                batch_size,
                tf.shape(self._betas)[0],
                f"{type(self).__name__} requires a fixed batch size. Got batch size {batch_size}"
                f" but previous batch size was {tf.shape(self._betas)[0]}.",
            )

        if not self._initialized:
            normal = tfp.distributions.Normal(
                tf.cast(0.0, dtype=x.dtype), tf.cast(1.0, dtype=x.dtype)
            )
            spread = 0.5 + 0.5 * tf.range(1, batch_size + 1, dtype=x.dtype) / (
                tf.cast(batch_size, dtype=x.dtype) + 1.0
            )  # [B]
            betas = normal.quantile(spread)  # [B]
            scaled_betas = 5.0 * tf.cast(self._search_space_dim, dtype=x.dtype) * betas  # [B]
            self._betas.assign(scaled_betas)  # [B]
            self._initialized.assign(True)

        mean, variance = self._model.predict(x)  # [..., B, 1]
        mean, variance = tf.squeeze(mean, -1), tf.squeeze(variance, -1)
        return -mean + tf.sqrt(variance) * self._betas  # [..., B]
