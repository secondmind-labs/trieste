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
This module contains multi-objective acquisition function builders.
"""
from __future__ import annotations

import math
from typing import Mapping, Optional, cast

import tensorflow as tf

from ...data import Dataset
from ...models import ProbabilisticModel
from ...observer import OBJECTIVE
from ...types import TensorType
from ..interface import (
    AcquisitionFunction,
    AcquisitionFunctionBuilder,
    GreedyAcquisitionFunctionBuilder,
    PenalizationFunction,
    ProbabilisticModelType,
    SingleModelAcquisitionBuilder,
)
from .hypervolume import ExpectedHypervolumeImprovement


class HIPPO(GreedyAcquisitionFunctionBuilder[ProbabilisticModelType]):
    r"""
    HIPPO: HIghly Parallelizable Pareto Optimization

    Builder of the acquisition function for greedily collecting batches by HIPPO
    penalization in multi-objective optimization by penalizing batch points
    by their distance in the objective space. The resulting acquistion function
    takes in a set of pending points and returns a base multi-objective acquisition function
    penalized around those points.

    Penalization is applied to the acquisition function multiplicatively. However, to
    improve numerical stability, we perform additive penalization in a log space.
    """

    def __init__(
        self,
        objective_tag: str = OBJECTIVE,
        base_acquisition_function_builder: AcquisitionFunctionBuilder[ProbabilisticModelType]
        | SingleModelAcquisitionBuilder[ProbabilisticModelType]
        | None = None,
    ):
        """
        Initializes the HIPPO acquisition function builder.

        :param objective_tag: The tag for the objective data and model.
        :param base_acquisition_function_builder: Base acquisition function to be
            penalized. Defaults to Expected Hypervolume Improvement, also supports
            its constrained version.
        """
        self._objective_tag = objective_tag
        if base_acquisition_function_builder is None:
            self._base_builder: AcquisitionFunctionBuilder[
                ProbabilisticModelType
            ] = ExpectedHypervolumeImprovement().using(self._objective_tag)
        else:
            if isinstance(base_acquisition_function_builder, SingleModelAcquisitionBuilder):
                self._base_builder = base_acquisition_function_builder.using(self._objective_tag)
            else:
                self._base_builder = base_acquisition_function_builder

        self._base_acquisition_function: Optional[AcquisitionFunction] = None
        self._penalization: Optional[PenalizationFunction] = None
        self._penalized_acquisition: Optional[AcquisitionFunction] = None

    def prepare_acquisition_function(
        self,
        models: Mapping[str, ProbabilisticModelType],
        datasets: Optional[Mapping[str, Dataset]] = None,
        pending_points: Optional[TensorType] = None,
    ) -> AcquisitionFunction:
        """
        Creates a new instance of the acquisition function.

        :param models: The models.
        :param datasets: The data from the observer. Must be populated.
        :param pending_points: The points we penalize with respect to.
        :return: The HIPPO acquisition function.
        :raise tf.errors.InvalidArgumentError: If the ``dataset`` is empty.
        """
        tf.debugging.Assert(datasets is not None, [])
        datasets = cast(Mapping[str, Dataset], datasets)
        tf.debugging.Assert(datasets[self._objective_tag] is not None, [])
        tf.debugging.assert_positive(
            len(datasets[self._objective_tag]),
            message=f"{self._objective_tag} dataset must be populated.",
        )

        acq = self._update_base_acquisition_function(models, datasets)
        if pending_points is not None and len(pending_points) != 0:
            acq = self._update_penalization(acq, models[self._objective_tag], pending_points)

        return acq

    def update_acquisition_function(
        self,
        function: AcquisitionFunction,
        models: Mapping[str, ProbabilisticModelType],
        datasets: Optional[Mapping[str, Dataset]] = None,
        pending_points: Optional[TensorType] = None,
        new_optimization_step: bool = True,
    ) -> AcquisitionFunction:
        """
        Updates the acquisition function.

        :param function: The acquisition function to update.
        :param models: The models.
        :param datasets: The data from the observer. Must be populated.
        :param pending_points: Points already chosen to be in the current batch (of shape [M,D]),
            where M is the number of pending points and D is the search space dimension.
        :param new_optimization_step: Indicates whether this call to update_acquisition_function
            is to start of a new optimization step, of to continue collecting batch of points
            for the current step. Defaults to ``True``.
        :return: The updated acquisition function.
        """
        tf.debugging.Assert(datasets is not None, [])
        datasets = cast(Mapping[str, Dataset], datasets)
        tf.debugging.Assert(datasets[self._objective_tag] is not None, [])
        tf.debugging.assert_positive(
            len(datasets[self._objective_tag]),
            message=f"{self._objective_tag} dataset must be populated.",
        )
        tf.debugging.Assert(self._base_acquisition_function is not None, [])

        if new_optimization_step:
            self._update_base_acquisition_function(models, datasets)

        if pending_points is None or len(pending_points) == 0:
            # no penalization required if no pending_points
            return cast(AcquisitionFunction, self._base_acquisition_function)

        return self._update_penalization(function, models[self._objective_tag], pending_points)

    def _update_penalization(
        self,
        function: AcquisitionFunction,
        model: ProbabilisticModel,
        pending_points: Optional[TensorType] = None,
    ) -> AcquisitionFunction:
        tf.debugging.assert_rank(pending_points, 2)

        if self._penalized_acquisition is not None and isinstance(
            self._penalization, hippo_penalizer
        ):
            # if possible, just update the penalization function variables
            # (the type ignore is due to mypy getting confused by tf.function)
            self._penalization.update(pending_points)  # type: ignore[unreachable]
            return self._penalized_acquisition
        else:
            # otherwise construct a new penalized acquisition function
            self._penalization = hippo_penalizer(model, pending_points)

        @tf.function
        def penalized_acquisition(x: TensorType) -> TensorType:
            log_acq = tf.math.log(
                cast(AcquisitionFunction, self._base_acquisition_function)(x)
            ) + tf.math.log(cast(PenalizationFunction, self._penalization)(x))
            return tf.math.exp(log_acq)

        self._penalized_acquisition = penalized_acquisition
        return penalized_acquisition

    def _update_base_acquisition_function(
        self,
        models: Mapping[str, ProbabilisticModelType],
        datasets: Optional[Mapping[str, Dataset]] = None,
    ) -> AcquisitionFunction:
        if self._base_acquisition_function is None:
            self._base_acquisition_function = self._base_builder.prepare_acquisition_function(
                models, datasets
            )
        else:
            self._base_acquisition_function = self._base_builder.update_acquisition_function(
                self._base_acquisition_function, models, datasets
            )
        return self._base_acquisition_function


class hippo_penalizer:
    r"""
    Returns the penalization function used for multi-objective greedy batch Bayesian
    optimization.

    A candidate point :math:`x` is penalized based on the Mahalanobis distance to a
    given pending point :math:`p_i`. Since we assume objectives to be independent,
    the Mahalanobis distance between these points becomes a Eucledian distance
    normalized by standard deviation. Penalties for multiple pending points are multiplied,
    and the resulting quantity is warped with the arctan function to :math:`[0, 1]` interval.

    :param model: The model over the specified ``dataset``.
    :param pending_points: The points we penalize with respect to.
    :return: The penalization function. This function will raise
        :exc:`ValueError` or :exc:`~tf.errors.InvalidArgumentError` if used with a batch size
        greater than one.
    """

    def __init__(self, model: ProbabilisticModel, pending_points: TensorType):
        """Initialize the MO penalizer.

        :param model: The model.
        :param pending_points: The points we penalize with respect to.
        :raise ValueError: If pending points are empty or None.
        :return: The penalization function. This function will raise
            :exc:`ValueError` or :exc:`~tf.errors.InvalidArgumentError` if used with a batch size
            greater than one."""
        tf.debugging.Assert(pending_points is not None and len(pending_points) != 0, [])

        self._model = model
        self._pending_points = tf.Variable(pending_points, shape=[None, *pending_points.shape[1:]])
        pending_means, pending_vars = self._model.predict(self._pending_points)
        self._pending_means = tf.Variable(pending_means, shape=[None, *pending_means.shape[1:]])
        self._pending_vars = tf.Variable(pending_vars, shape=[None, *pending_vars.shape[1:]])

    def update(self, pending_points: TensorType) -> None:
        """Update the penalizer with new pending points."""
        tf.debugging.Assert(pending_points is not None and len(pending_points) != 0, [])

        self._pending_points.assign(pending_points)
        pending_means, pending_vars = self._model.predict(self._pending_points)
        self._pending_means.assign(pending_means)
        self._pending_vars.assign(pending_vars)

    @tf.function
    def __call__(self, x: TensorType) -> TensorType:
        tf.debugging.assert_shapes(
            [(x, [..., 1, None])],
            message="This penalization function cannot be calculated for batches of points.",
        )

        # x is [N, 1, D]
        x = tf.squeeze(x, axis=1)  # x is now [N, D]

        x_means, x_vars = self._model.predict(x)
        # x_means is [N, K], x_vars is [N, K]
        # where K is the number of models/objectives

        # self._pending_points is [B, D] where B is the size of the batch collected so far
        tf.debugging.assert_shapes(
            [
                (x, ["N", "D"]),
                (self._pending_points, ["B", "D"]),
                (self._pending_means, ["B", "K"]),
                (self._pending_vars, ["B", "K"]),
                (x_means, ["N", "K"]),
                (x_vars, ["N", "K"]),
            ],
            message="""Encountered unexpected shapes while calculating mean and variance
                       of given point x and pending points""",
        )

        x_means_expanded = x_means[:, None, :]
        pending_means_expanded = self._pending_means[None, :, :]
        pending_vars_expanded = self._pending_vars[None, :, :]
        pending_stddevs_expanded = tf.sqrt(pending_vars_expanded)

        # this computes Mahalanobis distance between x and pending points
        # since we assume objectives to be independent
        # it reduces to regular Eucledian distance normalized by standard deviation
        standardize_mean_diff = (
            tf.abs(x_means_expanded - pending_means_expanded) / pending_stddevs_expanded
        )  # [N, B, K]
        d = tf.norm(standardize_mean_diff, axis=-1)  # [N, B]

        # warp the distance so that resulting value is from 0 to (nearly) 1
        warped_d = (2.0 / math.pi) * tf.math.atan(d)
        penalty = tf.reduce_prod(warped_d, axis=-1)  # [N,]

        return tf.reshape(penalty, (-1, 1))
