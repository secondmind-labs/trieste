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

from typing import Callable, Dict, Mapping, Optional, Union, cast

import gpflow
import tensorflow as tf
import tensorflow_probability as tfp
from check_shapes import check_shapes
from typing_extensions import Protocol, runtime_checkable

from ...data import Dataset
from ...models import FastUpdateModel, ModelStack, ProbabilisticModel
from ...models.interfaces import (
    PredictJointModelStack,
    PredictJointPredictYModelStack,
    PredictYModelStack,
    SupportsGetKernel,
    SupportsGetObservationNoise,
    SupportsPredictJoint,
    SupportsPredictY,
)
from ...observer import OBJECTIVE
from ...space import SearchSpace
from ...types import Tag, TensorType
from ..interface import (
    AcquisitionFunction,
    AcquisitionFunctionBuilder,
    GreedyAcquisitionFunctionBuilder,
    PenalizationFunction,
    SingleModelAcquisitionBuilder,
    SingleModelGreedyAcquisitionBuilder,
    UpdatablePenalizationFunction,
)
from .entropy import MinValueEntropySearch
from .function import ExpectedImprovement, MakePositive, expected_improvement


class LocalPenalization(SingleModelGreedyAcquisitionBuilder[ProbabilisticModel]):
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
        penalizer: Optional[
            Callable[
                [ProbabilisticModel, TensorType, TensorType, TensorType],
                Union[PenalizationFunction, UpdatablePenalizationFunction],
            ]
        ] = None,
        base_acquisition_function_builder: (
            ExpectedImprovement
            | MinValueEntropySearch[ProbabilisticModel]
            | MakePositive[ProbabilisticModel]
            | None
        ) = None,
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
            self._base_builder: SingleModelAcquisitionBuilder[ProbabilisticModel] = (
                ExpectedImprovement()
            )
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
        tf.debugging.Assert(dataset is not None, [tf.constant([])])
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
        tf.debugging.Assert(dataset is not None, [tf.constant([])])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")
        tf.debugging.Assert(self._base_acquisition_function is not None, [tf.constant([])])

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
            self._penalized_acquisition = PenalizedAcquisition(
                cast(PenalizedAcquisition, self._base_acquisition_function), self._penalization
            )
            return self._penalized_acquisition

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


class PenalizedAcquisition:
    """Class representing a penalized acquisition function."""

    # (note that this needs to be defined as a top level class make it pickleable)
    def __init__(
        self, base_acquisition_function: AcquisitionFunction, penalization: PenalizationFunction
    ):
        """
        :param base_acquisition_function: Base (unpenalized) acquisition function.
        :param penalization: Penalization function.
        """
        self._base_acquisition_function = base_acquisition_function
        self._penalization = penalization

    @tf.function
    def __call__(self, x: TensorType) -> TensorType:
        log_acq = tf.math.log(self._base_acquisition_function(x)) + tf.math.log(
            self._penalization(x)
        )
        return tf.math.exp(log_acq)


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

        normal = tfp.distributions.Normal(tf.constant(0, x.dtype), tf.constant(1, x.dtype))
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


@runtime_checkable
class FantasizerModelType(
    FastUpdateModel,
    SupportsPredictJoint,
    SupportsPredictY,
    SupportsGetKernel,
    SupportsGetObservationNoise,
    Protocol,
):
    """The model requirements for the Fantasizer acquisition function."""


class FantasizerModelStack(
    PredictJointModelStack, PredictYModelStack, ModelStack[FantasizerModelType]
):
    """
    A stack of models :class:`FantasizerModelType` models. Note that this delegates predict_joint
    and predict_y but none of the other methods.
    """


FantasizerModelOrStack = Union[FantasizerModelType, FantasizerModelStack]


class Fantasizer(GreedyAcquisitionFunctionBuilder[FantasizerModelOrStack]):
    r"""
    Builder of the acquisition function maker for greedily collecting batches.
    Fantasizer allows us to perform batch Bayesian optimization with any
    standard (non-batch) acquisition function.

    Here, every time a query point is chosen by maximising an acquisition function,
    its corresponding observation is "fantasized", and the models are conditioned further
    on this new artificial data.

    This implies that the models need to predict what their updated predictions would be given
    new data, see :class:`~FastUpdateModel`. These equations are for instance in closed form
    for the GPR model, see :cite:`chevalier2014corrected` (eqs. 8-10) for details.

    There are several ways to "fantasize" data: the "kriging believer" heuristic (KB, see
    :cite:`ginsbourger2010kriging`) uses the mean of the model as observations.
    "sample" uses samples from the model.
    """

    def __init__(
        self,
        base_acquisition_function_builder: Optional[
            AcquisitionFunctionBuilder[SupportsPredictJoint]
            | SingleModelAcquisitionBuilder[SupportsPredictJoint]
        ] = None,
        fantasize_method: str = "KB",
    ):
        """

        :param base_acquisition_function_builder: The acquisition function builder to use.
            Defaults to :class:`~trieste.acquisition.ExpectedImprovement`.
        :param fantasize_method: The following options are available: "KB" and "sample".
            See class docs for more details.
        :raise tf.errors.InvalidArgumentError: If ``fantasize_method`` is not "KB" or "sample".
        """
        tf.debugging.Assert(fantasize_method in ["KB", "sample"], [tf.constant([])])

        if base_acquisition_function_builder is None:
            base_acquisition_function_builder = ExpectedImprovement()

        if isinstance(base_acquisition_function_builder, SingleModelAcquisitionBuilder):
            base_acquisition_function_builder = base_acquisition_function_builder.using(OBJECTIVE)

        self._builder = base_acquisition_function_builder
        self._fantasize_method = fantasize_method

        self._base_acquisition_function: Optional[AcquisitionFunction] = None
        self._fantasized_acquisition: Optional[AcquisitionFunction] = None
        self._fantasized_models: Mapping[
            Tag, _fantasized_model | ModelStack[SupportsPredictJoint]
        ] = {}

    def _update_base_acquisition_function(
        self,
        models: Mapping[Tag, FantasizerModelOrStack],
        datasets: Optional[Mapping[Tag, Dataset]],
    ) -> AcquisitionFunction:
        if self._base_acquisition_function is not None:
            self._base_acquisition_function = self._builder.update_acquisition_function(
                self._base_acquisition_function, models, datasets
            )
        else:
            self._base_acquisition_function = self._builder.prepare_acquisition_function(
                models, datasets
            )
        return self._base_acquisition_function

    def _update_fantasized_acquisition_function(
        self,
        models: Mapping[Tag, FantasizerModelOrStack],
        datasets: Optional[Mapping[Tag, Dataset]],
        pending_points: TensorType,
    ) -> AcquisitionFunction:
        tf.debugging.assert_rank(pending_points, 2)

        fantasized_data = {
            tag: _generate_fantasized_data(
                fantasize_method=self._fantasize_method,
                model=model,
                pending_points=pending_points,
            )
            for tag, model in models.items()
        }

        if datasets is None:
            datasets = fantasized_data
        else:
            datasets = {tag: data + fantasized_data[tag] for tag, data in datasets.items()}

        if self._fantasized_acquisition is None:
            self._fantasized_models = {
                tag: _generate_fantasized_model(model, fantasized_data[tag])
                for tag, model in models.items()
            }
            self._fantasized_acquisition = self._builder.prepare_acquisition_function(
                cast(Dict[Tag, SupportsPredictJoint], self._fantasized_models), datasets
            )
        else:
            for tag, model in self._fantasized_models.items():
                if isinstance(model, ModelStack):
                    observations = tf.split(
                        fantasized_data[tag].observations, model._event_sizes, axis=-1
                    )
                    for submodel, obs in zip(model._models, observations):
                        submodel.update_fantasized_data(
                            Dataset(fantasized_data[tag].query_points, obs)
                        )

                else:
                    model.update_fantasized_data(fantasized_data[tag])
            self._builder.update_acquisition_function(
                self._fantasized_acquisition,
                cast(Dict[Tag, SupportsPredictJoint], self._fantasized_models),
                datasets,
            )

        return self._fantasized_acquisition

    def prepare_acquisition_function(
        self,
        models: Mapping[Tag, FantasizerModelOrStack],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
        pending_points: Optional[TensorType] = None,
    ) -> AcquisitionFunction:
        """

        :param models: The models over each tag.
        :param datasets: The data from the observer (optional).
        :param pending_points: Points already chosen to be in the current batch (of shape [M,D]),
            where M is the number of pending points and D is the search space dimension.
        :return: An acquisition function.
        """
        for model in models.values():
            if not (
                isinstance(model, FantasizerModelType)
                or isinstance(model, ModelStack)
                and all(isinstance(m, FantasizerModelType) for m in model._models)
            ):
                raise NotImplementedError(
                    f"Fantasizer only works with FastUpdateModel models that also support "
                    f"predict_joint, get_kernel and get_observation_noise, or with "
                    f"ModelStack stacks of such models; received {model!r}"
                )
        if pending_points is None:
            return self._update_base_acquisition_function(models, datasets)
        else:
            return self._update_fantasized_acquisition_function(models, datasets, pending_points)

    def update_acquisition_function(
        self,
        function: AcquisitionFunction,
        models: Mapping[Tag, FantasizerModelOrStack],
        datasets: Optional[Mapping[Tag, Dataset]] = None,
        pending_points: Optional[TensorType] = None,
        new_optimization_step: bool = True,
    ) -> AcquisitionFunction:
        """
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
        if pending_points is None:
            return self._update_base_acquisition_function(models, datasets)
        else:
            return self._update_fantasized_acquisition_function(models, datasets, pending_points)


def _generate_fantasized_data(
    fantasize_method: str, model: FantasizerModelOrStack, pending_points: TensorType
) -> Dataset:
    """
    Generates "fantasized" data at pending_points depending on the chosen heuristic:
    - KB (kriging believer) uses the mean prediction of the models
    - sample uses samples from the GP posterior.

    :param fantasize_method: the following options are available: "KB" and "sample".
    :param model: a model with predict method
    :param dataset: past data
    :param pending_points: points at which to fantasize data
    :return: a fantasized dataset
    """
    if fantasize_method == "KB":
        fantasized_obs, _ = model.predict(pending_points)
    elif fantasize_method == "sample":
        fantasized_obs = model.sample(pending_points, num_samples=1)[0]
    else:
        raise NotImplementedError(f"fantasize_method must be KB or sample, received {model!r}")

    return Dataset(pending_points, fantasized_obs)


def _generate_fantasized_model(
    model: FantasizerModelOrStack, fantasized_data: Dataset
) -> _fantasized_model | PredictJointPredictYModelStack:
    if isinstance(model, ModelStack):
        observations = tf.split(fantasized_data.observations, model._event_sizes, axis=-1)
        fmods = []
        for mod, obs, event_size in zip(model._models, observations, model._event_sizes):
            fmods.append(
                (
                    _fantasized_model(mod, Dataset(fantasized_data.query_points, obs)),
                    event_size,
                )
            )
        return PredictJointPredictYModelStack(*fmods)
    else:
        return _fantasized_model(model, fantasized_data)


class _fantasized_model(
    SupportsPredictJoint, SupportsGetKernel, SupportsGetObservationNoise, SupportsPredictY
):
    """
    Creates a new model from an existing one and additional data.
    This new model posterior is conditioned on both current model data and the additional one.
    """

    def __init__(self, model: FantasizerModelType, fantasized_data: Dataset):
        """
        :param model: a model, must be of class `FastUpdateModel`
        :param fantasized_data: additional dataset to condition on
        :raise NotImplementedError: If model is not of class `FastUpdateModel`.
        """

        self._model = model
        self._fantasized_query_points = tf.Variable(
            fantasized_data.query_points,
            trainable=False,
            shape=[None, *fantasized_data.query_points.shape[1:]],
        )
        self._fantasized_observations = tf.Variable(
            fantasized_data.observations,
            trainable=False,
            shape=[None, *fantasized_data.observations.shape[1:]],
        )

    def update_fantasized_data(self, fantasized_data: Dataset) -> None:
        """
        :param fantasized_data: new additional dataset to condition on
        """
        self._fantasized_query_points.assign(fantasized_data.query_points)
        self._fantasized_observations.assign(fantasized_data.observations)

    @check_shapes(
        "query_points: [batch..., N, D]",
        "return[0]: [batch..., ..., N, L]",
        "return[1]: [batch..., ..., N, L]",
    )
    def predict(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        """
        This function wraps conditional_predict_f. It cannot directly call
        conditional_predict_f, since it does not accept query_points with rank > 2.
        We use map_fn to allow leading dimensions for query_points.

        :param query_points: shape [...*, N, d]
        :return: mean, shape [...*, ..., N, L] and cov, shape [...*, ..., N, L],
            where ... are the leading dimensions of fantasized_data
        """

        def fun(qp: TensorType) -> tuple[TensorType, TensorType]:  # pragma: no cover (tf.map_fn)
            fantasized_data = Dataset(
                self._fantasized_query_points.value(), self._fantasized_observations.value()
            )
            return self._model.conditional_predict_f(qp, fantasized_data)

        return _broadcast_predict(query_points, fun)

    @check_shapes(
        "query_points: [batch..., N, D]",
        "return[0]: [batch..., ..., N, L]",
        "return[1]: [batch..., ..., L, N, N]",
    )
    def predict_joint(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        """
        This function wraps conditional_predict_joint. It cannot directly call
        conditional_predict_joint, since it does not accept query_points with rank > 2.
        We use map_fn to allow leading dimensions for query_points.

        :param query_points: shape [...*, N, D]
        :return: mean, shape [...*, ..., N, L] and cov, shape [...*, ..., L, N, N],
            where ... are the leading dimensions of fantasized_data
        """

        def fun(qp: TensorType) -> tuple[TensorType, TensorType]:  # pragma: no cover (tf.map_fn)
            fantasized_data = Dataset(
                self._fantasized_query_points.value(), self._fantasized_observations.value()
            )
            return self._model.conditional_predict_joint(qp, fantasized_data)

        return _broadcast_predict(query_points, fun)

    @check_shapes(
        "query_points: [batch..., N, D]",
        "return: [batch..., ..., S, N, L]",
    )
    def sample(self, query_points: TensorType, num_samples: int) -> TensorType:
        """
        This function wraps conditional_predict_f_sample. It cannot directly call
        conditional_predict_joint, since it does not accept query_points with rank > 2.
        We use map_fn to allow leading dimensions for query_points.

        :param query_points: shape [...*, N, D]
        :param num_samples: number of samples.
        :return: samples of shape [...*, ..., S, N, L], where ... are the leading
            dimensions of fantasized_data
        """
        leading_dim, query_points_flatten = _get_leading_dim_and_flatten(query_points)
        # query_points_flatten: [B, n, d], leading_dim =...*, product = B

        samples = tf.map_fn(
            fn=lambda qp: self._model.conditional_predict_f_sample(
                qp,
                Dataset(
                    self._fantasized_query_points.value(), self._fantasized_observations.value()
                ),
                num_samples,
            ),
            elems=query_points_flatten,
        )  # [B, ..., S, L]
        return _restore_leading_dim(samples, leading_dim)

    @check_shapes(
        "query_points: [broadcast batch..., N, D]",
        "return[0]: [batch..., ..., N, L]",
        "return[1]: [batch..., ..., N, L]",
    )
    def predict_y(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        """
        This function wraps conditional_predict_y. It cannot directly call
        conditional_predict_joint, since it does not accept query_points with rank > 2.
        We use tf.map_fn to allow leading dimensions for query_points.

        :param query_points: shape [...*, N, D]
        :return: mean, shape [...*, ..., N, L] and var, shape [...*, ..., N, L],
            where ... are the leading dimensions of fantasized_data
        """

        def fun(qp: TensorType) -> tuple[TensorType, TensorType]:  # pragma: no cover (tf.map_fn)
            fantasized_data = Dataset(
                self._fantasized_query_points.value(), self._fantasized_observations.value()
            )
            return self._model.conditional_predict_y(qp, fantasized_data)

        return _broadcast_predict(query_points, fun)

    def get_observation_noise(self) -> TensorType:
        return self._model.get_observation_noise()

    def get_kernel(self) -> gpflow.kernels.Kernel:
        return self._model.get_kernel()

    def log(self, dataset: Optional[Dataset] = None) -> None:
        return self._model.log(dataset)


def _broadcast_predict(
    query_points: TensorType, fun: Callable[[TensorType], tuple[TensorType, TensorType]]
) -> tuple[TensorType, TensorType]:
    """
    Utility function that allows leading dimensions for query_points when
    fun only accepts rank 2 tensors. It works by flattening query_points into
    a rank 3 tensor, evaluate fun(query_points) through tf.map_fn, then
    restoring the leading dimensions.

    :param query_points: shape [...*, N, D]
    :param fun: callable that returns two tensors (e.g. a predict function)
    :return: two tensors (e.g. mean and variance) with shape [...*, ...]
    """

    leading_dim, query_points_flatten = _get_leading_dim_and_flatten(query_points)
    # leading_dim =...*, product = B
    # query_points_flatten: [B, N, D]

    mean_signature = tf.TensorSpec(None, query_points.dtype)
    var_signature = tf.TensorSpec(None, query_points.dtype)
    mean, var = tf.map_fn(
        fn=fun,
        elems=query_points_flatten,
        fn_output_signature=(mean_signature, var_signature),
    )  # [B, ..., L, N], [B, ..., L, N] (predict_f) or [B, ..., L, N, N] (predict_joint)

    return _restore_leading_dim(mean, leading_dim), _restore_leading_dim(var, leading_dim)


def _get_leading_dim_and_flatten(query_points: TensorType) -> tuple[TensorType, TensorType]:
    """
    :param query_points: shape [...*, N, D]
    :return: leading_dim = ....*, query_points_flatten, shape [B, N, D]
    """
    leading_dim = tf.shape(query_points)[:-2]  # =...*, product = B
    nd = tf.shape(query_points)[-2:]
    query_points_flatten = tf.reshape(query_points, (-1, nd[0], nd[1]))  # [B, N, D]
    return leading_dim, query_points_flatten


def _restore_leading_dim(x: TensorType, leading_dim: TensorType) -> TensorType:
    """
    "Un-flatten" the first dimension of x to leading_dim

    :param x: shape [B, ...]
    :param leading_dim: [...*]
    :return: shape [...*, ...]
    """
    single_x_shape = tf.shape(x[0])  # = [...]
    output_x_shape = tf.concat([leading_dim, single_x_shape], axis=0)  # = [...*, ...]
    return tf.reshape(x, output_x_shape)  # [...*, ...]
