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
This module contains local penalization-based acquisition function builders.
"""
from __future__ import annotations

from typing import Callable, Optional, Union, cast

import tensorflow as tf
import tensorflow_probability as tfp

from ...data import Dataset
from ...models import ProbabilisticModel
from ...space import SearchSpace
from ...types import TensorType
from ..interface import (
    AcquisitionFunction,
    PenalizationFunction,
    SingleModelAcquisitionBuilder,
    SingleModelGreedyAcquisitionBuilder,
    UpdatablePenalizationFunction,
)
from .entropy import MinValueEntropySearch
from .function import ExpectedImprovement, expected_improvement


class LocalPenalizationAcquisitionFunction(SingleModelGreedyAcquisitionBuilder):
    r"""
    Builder of the acquisition function maker for greedily collecting batches by local
    penalization.  The resulting :const:`AcquisitionFunctionMaker` takes in a set of pending
    points and returns a base acquisition function penalized around those points.
    An estimate of the objective function's Lipschitz constant is used to control the size
    of penalization.

    Local penalization allows us to perform batch Bayesian optimization with a standard (non-batch)
    acquisition function. All that we require is that the acquisition function takes strictly
    positive values. By iteratively building a batch of points though sequentially maximizing
    this acquisition function but down-weighted around locations close to the already
    chosen (pending) points, local penalization provides diverse batches of candidate points.

    Local penalization is applied to the acquisition function multiplicatively. However, to
    improve numerical stability, we perform additive penalization in a log space.

    The Lipschitz constant and additional penalization parameters are estimated once
    when first preparing the acquisition function with no pending points. These estimates
    are reused for all subsequent function calls.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        num_samples: int = 500,
        penalizer: Callable[
            [ProbabilisticModel, TensorType, TensorType, TensorType],
            Union[PenalizationFunction, UpdatablePenalizationFunction],
        ] = None,
        base_acquisition_function_builder: ExpectedImprovement
        | MinValueEntropySearch
        | None = None,
    ):
        """
        :param search_space: The global search space over which the optimisation is defined.
        :param num_samples: Size of the random sample over which the Lipschitz constant
            is estimated. We recommend scaling this with search space dimension.
        :param penalizer: The chosen penalization method (defaults to soft penalization). This
            should be a function that accepts a model, pending points, lipschitz constant and eta
            and returns a PenalizationFunction.
        :param base_acquisition_function_builder: Base acquisition function to be
            penalized (defaults to expected improvement). Local penalization only supports
            strictly positive acquisition functions.
        :raise tf.errors.InvalidArgumentError: If ``num_samples`` is not positive.
        """
        tf.debugging.assert_positive(num_samples)

        self._search_space = search_space
        self._num_samples = num_samples

        self._lipschitz_penalizer = soft_local_penalizer if penalizer is None else penalizer

        if base_acquisition_function_builder is None:
            self._base_builder: SingleModelAcquisitionBuilder = ExpectedImprovement()
        else:
            self._base_builder = base_acquisition_function_builder

        self._lipschitz_constant = None
        self._eta = None
        self._base_acquisition_function: Optional[AcquisitionFunction] = None
        self._penalization: Optional[PenalizationFunction | UpdatablePenalizationFunction] = None
        self._penalized_acquisition: Optional[AcquisitionFunction] = None

    def prepare_acquisition_function(
        self,
        model: ProbabilisticModel,
        dataset: Optional[Dataset] = None,
        pending_points: Optional[TensorType] = None,
    ) -> AcquisitionFunction:
        """
        :param model: The model.
        :param dataset: The data from the observer. Must be populated.
        :param pending_points: The points we penalize with respect to.
        :return: The (log) expected improvement penalized with respect to the pending points.
        :raise tf.errors.InvalidArgumentError: If the ``dataset`` is empty.
        """
        tf.debugging.Assert(dataset is not None, [])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")

        acq = self._update_base_acquisition_function(dataset, model)
        if pending_points is not None and len(pending_points) != 0:
            acq = self._update_penalization(acq, dataset, model, pending_points)

        return acq

    def update_acquisition_function(
        self,
        function: AcquisitionFunction,
        model: ProbabilisticModel,
        dataset: Optional[Dataset] = None,
        pending_points: Optional[TensorType] = None,
        new_optimization_step: bool = True,
    ) -> AcquisitionFunction:
        """
        :param function: The acquisition function to update.
        :param model: The model.
        :param dataset: The data from the observer. Must be populated.
        :param pending_points: Points already chosen to be in the current batch (of shape [M,D]),
            where M is the number of pending points and D is the search space dimension.
        :param new_optimization_step: Indicates whether this call to update_acquisition_function
            is to start of a new optimization step, of to continue collecting batch of points
            for the current step. Defaults to ``True``.
        :return: The updated acquisition function.
        """
        tf.debugging.Assert(dataset is not None, [])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")
        tf.debugging.Assert(self._base_acquisition_function is not None, [])

        if new_optimization_step:
            self._update_base_acquisition_function(dataset, model)

        if pending_points is None or len(pending_points) == 0:
            # no penalization required if no pending_points
            return cast(AcquisitionFunction, self._base_acquisition_function)

        return self._update_penalization(function, dataset, model, pending_points)

    def _update_penalization(
        self,
        function: Optional[AcquisitionFunction],
        dataset: Dataset,
        model: ProbabilisticModel,
        pending_points: Optional[TensorType] = None,
    ) -> AcquisitionFunction:
        tf.debugging.assert_rank(pending_points, 2)

        if self._penalized_acquisition is not None and isinstance(
            self._penalization, UpdatablePenalizationFunction
        ):
            # if possible, just update the penalization function variables
            self._penalization.update(pending_points, self._lipschitz_constant, self._eta)
            return self._penalized_acquisition
        else:
            # otherwise construct a new penalized acquisition function
            self._penalization = self._lipschitz_penalizer(
                model, pending_points, self._lipschitz_constant, self._eta
            )

            @tf.function
            def penalized_acquisition(x: TensorType) -> TensorType:
                log_acq = tf.math.log(
                    cast(AcquisitionFunction, self._base_acquisition_function)(x)
                ) + tf.math.log(cast(PenalizationFunction, self._penalization)(x))
                return tf.math.exp(log_acq)

            self._penalized_acquisition = penalized_acquisition
            return penalized_acquisition

    @tf.function(experimental_relax_shapes=True)
    def _get_lipschitz_estimate(
        self, model: ProbabilisticModel, sampled_points: TensorType
    ) -> tuple[TensorType, TensorType]:
        with tf.GradientTape() as g:
            g.watch(sampled_points)
            mean, _ = model.predict(sampled_points)
        grads = g.gradient(mean, sampled_points)
        grads_norm = tf.norm(grads, axis=1)
        max_grads_norm = tf.reduce_max(grads_norm)
        eta = tf.reduce_min(mean, axis=0)
        return max_grads_norm, eta

    def _update_base_acquisition_function(
        self, dataset: Dataset, model: ProbabilisticModel
    ) -> AcquisitionFunction:
        samples = self._search_space.sample(num_samples=self._num_samples)
        samples = tf.concat([dataset.query_points, samples], 0)

        lipschitz_constant, eta = self._get_lipschitz_estimate(model, samples)
        if lipschitz_constant < 1e-5:  # threshold to improve numerical stability for 'flat' models
            lipschitz_constant = 10

        self._lipschitz_constant = lipschitz_constant
        self._eta = eta

        if self._base_acquisition_function is not None:
            self._base_acquisition_function = self._base_builder.update_acquisition_function(
                self._base_acquisition_function,
                model,
                dataset=dataset,
            )
        elif isinstance(self._base_builder, ExpectedImprovement):  # reuse eta estimate
            self._base_acquisition_function = cast(
                AcquisitionFunction, expected_improvement(model, self._eta)
            )
        else:
            self._base_acquisition_function = self._base_builder.prepare_acquisition_function(
                model,
                dataset=dataset,
            )
        return self._base_acquisition_function


class local_penalizer(UpdatablePenalizationFunction):
    def __init__(
        self,
        model: ProbabilisticModel,
        pending_points: TensorType,
        lipschitz_constant: TensorType,
        eta: TensorType,
    ):
        """Initialize the local penalizer.

        :param model: The model over the specified ``dataset``.
        :param pending_points: The points we penalize with respect to.
        :param lipschitz_constant: The estimated Lipschitz constant of the objective function.
        :param eta: The estimated global minima.
        :return: The local penalization function. This function will raise
            :exc:`ValueError` or :exc:`~tf.errors.InvalidArgumentError` if used with a batch size
            greater than one."""
        self._model = model

        mean_pending, variance_pending = model.predict(pending_points)
        self._pending_points = tf.Variable(pending_points, shape=[None, *pending_points.shape[1:]])
        self._radius = tf.Variable(
            tf.transpose((mean_pending - eta) / lipschitz_constant),
            shape=[1, None],
        )
        self._scale = tf.Variable(
            tf.transpose(tf.sqrt(variance_pending) / lipschitz_constant),
            shape=[1, None],
        )

    def update(
        self,
        pending_points: TensorType,
        lipschitz_constant: TensorType,
        eta: TensorType,
    ) -> None:
        """Update the local penalizer with new variable values."""
        mean_pending, variance_pending = self._model.predict(pending_points)
        self._pending_points.assign(pending_points)
        self._radius.assign(tf.transpose((mean_pending - eta) / lipschitz_constant))
        self._scale.assign(tf.transpose(tf.sqrt(variance_pending) / lipschitz_constant))


class soft_local_penalizer(local_penalizer):

    r"""
    Return the soft local penalization function used for single-objective greedy batch Bayesian
    optimization in :cite:`Gonzalez:2016`.

    Soft penalization returns the probability that a candidate point does not belong
    in the exclusion zones of the pending points. For model posterior mean :math:`\mu`, model
    posterior variance :math:`\sigma^2`, current "best" function value :math:`\eta`, and an
    estimated Lipschitz constant :math:`L`,the penalization from a set of pending point
    :math:`x'` on a candidate point :math:`x` is given by
    .. math:: \phi(x, x') = \frac{1}{2}\textrm{erfc}(-z)
    where :math:`z = \frac{1}{\sqrt{2\sigma^2(x')}}(L||x'-x|| + \eta - \mu(x'))`.

    The penalization from a set of pending points is just product of the individual
    penalizations. See :cite:`Gonzalez:2016` for a full derivation.

    :param model: The model over the specified ``dataset``.
    :param pending_points: The points we penalize with respect to.
    :param lipschitz_constant: The estimated Lipschitz constant of the objective function.
    :param eta: The estimated global minima.
    :return: The local penalization function. This function will raise
        :exc:`ValueError` or :exc:`~tf.errors.InvalidArgumentError` if used with a batch size
        greater than one.
    """

    @tf.function
    def __call__(self, x: TensorType) -> TensorType:
        tf.debugging.assert_shapes(
            [(x, [..., 1, None])],
            message="This penalization function cannot be calculated for batches of points.",
        )

        pairwise_distances = tf.norm(
            tf.expand_dims(x, 1) - tf.expand_dims(self._pending_points, 0), axis=-1
        )
        standardised_distances = (pairwise_distances - self._radius) / self._scale

        normal = tfp.distributions.Normal(tf.cast(0, x.dtype), tf.cast(1, x.dtype))
        penalization = normal.cdf(standardised_distances)
        return tf.reduce_prod(penalization, axis=-1)


class hard_local_penalizer(local_penalizer):
    r"""
    Return the hard local penalization function used for single-objective greedy batch Bayesian
    optimization in :cite:`Alvi:2019`.

    Hard penalization is a stronger penalizer than soft penalization and is sometimes more effective
    See :cite:`Alvi:2019` for details. Our implementation follows theirs, with the penalization from
    a set of pending points being the product of the individual penalizations.

    :param model: The model over the specified ``dataset``.
    :param pending_points: The points we penalize with respect to.
    :param lipschitz_constant: The estimated Lipschitz constant of the objective function.
    :param eta: The estimated global minima.
    :return: The local penalization function. This function will raise
        :exc:`ValueError` or :exc:`~tf.errors.InvalidArgumentError` if used with a batch size
        greater than one.
    """

    @tf.function
    def __call__(self, x: TensorType) -> TensorType:
        tf.debugging.assert_shapes(
            [(x, [..., 1, None])],
            message="This penalization function cannot be calculated for batches of points.",
        )

        pairwise_distances = tf.norm(
            tf.expand_dims(x, 1) - tf.expand_dims(self._pending_points, 0), axis=-1
        )

        p = -5  # following experiments of :cite:`Alvi:2019`.
        penalization = ((pairwise_distances / (self._radius + self._scale)) ** p + 1) ** (1 / p)
        return tf.reduce_prod(penalization, axis=-1)
