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
This module is the home of  Trieste's functionality for choosing the inducing points
of sparse variational Gaussian processes (i.e. our :class:`SparseVariational` wrapper).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic

import gpflow
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.cluster.vq import kmeans

from ...data import Dataset
from ...space import Box, DiscreteSearchSpace, SearchSpace
from ...types import TensorType
from ..interfaces import ProbabilisticModelType
from .interface import GPflowPredictor


class InducingPointSelector(ABC, Generic[ProbabilisticModelType]):
    """
    This class provides functionality to update the inducing points of an inducing point-based model
    as the Bayesian optimization progresses.

    The only constraint on subclasses of :class:`InducingPointSelector` is that they preserve
    the shape of the inducing points so not to trigger expensive retracing.

    It can often be beneficial to change the inducing points during optimization, for example
    to allow the model to focus its limited modelling resources into promising areas of the space.
    See :cite:`vakili2021scalable` for demonstrations of some of
    our :class:`InducingPointSelectors`.
    """

    def __init__(self, recalc_every_model_update: bool = True):
        """
        :param recalc_every_model_update: If True then recalculate the inducing points for each
            model update, otherwise just recalculate on the first call.
        """
        self._recalc_every_model_update = recalc_every_model_update
        self._initialized = False

    def calculate_inducing_points(
        self,
        current_inducing_points: TensorType,
        model: ProbabilisticModelType,
        dataset: Dataset,
    ) -> TensorType:
        """
        Calculate the new inducing points given the existing inducing points.

        If `recalc_every_model_update` is set to False then we only generate new inducing points
        for the first :meth:`calculate_inducing_points` call, otherwise we just return the current
        inducing points.

        :param current_inducing_points: The current inducing points used by the model.
        :param model: The sparse model.
        :param dataset: The data from the observer.
        :return: The new updated inducing points.
        :raise NotImplementedError: If model has more than one set of inducing variables.
        """
        tf.debugging.Assert(current_inducing_points is not None, [tf.constant([])])

        if isinstance(current_inducing_points, list):
            raise NotImplementedError(
                """
                InducingPointSelectors only currently support models with a single set
                of inducing points.
                """
            )

        if (
            not self._initialized
        ) or self._recalc_every_model_update:  # calculate new inducing points when required
            self._initialized = True
            M = tf.shape(current_inducing_points)[0]
            new_inducing_points = self._recalculate_inducing_points(M, model, dataset)  # [M, D]
            tf.assert_equal(tf.shape(current_inducing_points), tf.shape(new_inducing_points))
            return new_inducing_points  # [M, D]
        else:  # otherwise dont recalculate
            return current_inducing_points  # [M, D]

    @abstractmethod
    def _recalculate_inducing_points(
        self, M: int, model: ProbabilisticModelType, dataset: Dataset
    ) -> TensorType:
        """
        Method for calculating new inducing points given a `model` and `dataset`.

        This method is to be implemented by all subclasses of :class:`InducingPointSelector`.

        :param M: Desired number of inducing points.
        :param model: The sparse model.
        :param dataset: The data from the observer.
        :return: The new updated inducing points.
        """
        raise NotImplementedError


class UniformInducingPointSelector(InducingPointSelector[GPflowPredictor]):
    """
    An :class:`InducingPointSelector` that chooses points sampled uniformly across the search space.
    """

    def __init__(self, search_space: SearchSpace, recalc_every_model_update: bool = True):
        """
        :param search_space: The global search space over which the optimization is defined.
        :param recalc_every_model_update: If True then recalculate the inducing points for each
            model update, otherwise just recalculate on the first call.
        """
        super().__init__(recalc_every_model_update)
        self._search_space = search_space

    def _recalculate_inducing_points(
        self, M: int, model: GPflowPredictor, dataset: Dataset
    ) -> TensorType:
        """
        Sample `M` points. If `search_space` is a :class:`Box` then we use a space-filling Sobol
        design to ensure high diversity.

        :param M: Desired number of inducing points.
        :param model: The sparse model .
        :param dataset: The data from the observer.
        :return: The new updated inducing points.
        """

        if isinstance(self._search_space, Box):
            return self._search_space.sample_sobol(M)
        else:
            return self._search_space.sample(M)


class RandomSubSampleInducingPointSelector(InducingPointSelector[GPflowPredictor]):
    """
    An :class:`InducingPointSelector` that chooses points at random from the training data.
    """

    def _recalculate_inducing_points(
        self, M: int, model: GPflowPredictor, dataset: Dataset
    ) -> TensorType:
        """
        Sample `M` points from the training data without replacement. If we require more
        inducing points than training data, then we fill the remaining points with random
        samples across the search space.

        :param M: Desired number of inducing points.
        :param model: The sparse model.
        :param dataset: The data from the observer. Must be populated.
        :return: The new updated inducing points.
        :raise tf.errors.InvalidArgumentError: If ``dataset`` is empty.
        """

        tf.debugging.Assert(len(dataset.query_points) is not None, [tf.constant([])])

        N = tf.shape(dataset.query_points)[0]  # training data size
        shuffled_query_points = tf.random.shuffle(dataset.query_points)  # [N, d]
        sub_sample = shuffled_query_points[: tf.math.minimum(N, M), :]

        if N < M:  # if fewer data than inducing points then sample remaining uniformly
            data_as_discrete_search_space = DiscreteSearchSpace(dataset.query_points)
            convex_hull_of_data = Box(
                lower=data_as_discrete_search_space.lower,
                upper=data_as_discrete_search_space.upper,
            )

            uniform_sampler = UniformInducingPointSelector(convex_hull_of_data)
            uniform_sample = uniform_sampler._recalculate_inducing_points(
                M - N, model, dataset
            )  # [M-N, d]
            sub_sample = tf.concat([sub_sample, uniform_sample], 0)  # [M, d]
        return sub_sample  # [M, d]


class KMeansInducingPointSelector(InducingPointSelector[GPflowPredictor]):
    """
    An :class:`InducingPointSelector` that chooses points as centroids of a K-means clustering
    of the training data.
    """

    def _recalculate_inducing_points(
        self, M: int, model: GPflowPredictor, dataset: Dataset
    ) -> TensorType:
        """
        Calculate `M` centroids from a K-means clustering of the training data.

        If the clustering returns fewer than `M` centroids or if we have fewer than `M` training
        data, then we fill the remaining points with random samples across the search space.

        :param M: Desired number of inducing points.
        :param model: The sparse model.
        :param dataset: The data from the observer. Must be populated.
        :return: The new updated inducing points.
        :raise tf.errors.InvalidArgumentError: If ``dataset`` is empty.
        """

        tf.debugging.Assert(dataset is not None, [tf.constant([])])

        query_points = dataset.query_points  # [N, d]
        N = tf.shape(query_points)[0]

        shuffled_query_points = tf.random.shuffle(query_points)  # [N, d]
        query_points_stds = tf.math.reduce_std(shuffled_query_points, 0)  # [d]

        if (
            tf.math.count_nonzero(query_points_stds, dtype=N.dtype) == N
        ):  # standardize if all stds non zero
            normalize = True
            shuffled_query_points = shuffled_query_points / query_points_stds  # [N, d]
        else:
            normalize = False

        centroids, _ = kmeans(shuffled_query_points, int(tf.math.minimum(M, N)))  # [C, d]

        if normalize:
            centroids *= query_points_stds  # [M, d]

        if len(centroids) < M:  # choose remaining points as random samples
            data_as_discrete_search_space = DiscreteSearchSpace(dataset.query_points)
            convex_hull_of_data = Box(
                lower=data_as_discrete_search_space.lower,
                upper=data_as_discrete_search_space.upper,
            )
            uniform_sampler = UniformInducingPointSelector(convex_hull_of_data)
            extra_centroids = uniform_sampler._recalculate_inducing_points(  # [M-C, d]
                M - len(centroids), model, dataset
            )
            centroids = tf.concat([centroids, extra_centroids], axis=0)  # [M, d]

        return centroids  # [M, d]


class QualityFunction(ABC):
    """
    A :const:`QualityFunction` uses a  `model` to measure the quality of each of
    the `N` query points in the provided `dataset`, returning shape `[N]`.
    """

    @abstractmethod
    def __call__(self, model: GPflowPredictor, dataset: Dataset) -> TensorType:
        """
        Evaluate the quality of the data-points according to the model.
        :param model: The sparse model.
        :param dataset: The data from the observer. Must be populated.
        :return: The quality scores.
        """


class DPPInducingPointSelector(InducingPointSelector[GPflowPredictor]):
    """
    An :class:`InducingPointSelector` that follows :cite:`chen2018fast` to get a greedy appoximation
    to the MAP estimate of the specified Determinantal Point Process (DPP).

    The DPP is defined through its diveristy-quality decomposition, i.e. its similarity kernel
    is just the kernel of the considered model and its quality scores come from the
    provided :class:`QualityFunction`.

    """

    def __init__(self, quality_function: QualityFunction, recalc_every_model_update: bool = True):
        """
        :param quality_function: A function measuring the quality of each candidate inducing point.
        :param recalc_every_model_update: If True then recalculate the inducing points for each
            model update, otherwise just recalculate on the first call.
        """
        super().__init__(recalc_every_model_update)
        self._quality_function = quality_function

    def _recalculate_inducing_points(
        self,
        M: int,
        model: GPflowPredictor,
        dataset: Dataset,
    ) -> TensorType:
        """
        :param M: Desired number of inducing points.
        :param model: The sparse model.
        :param dataset: The data from the observer. Must be populated.
        :return: The new updated inducing points.
        :raise tf.errors.InvalidArgumentError: If ``dataset`` is empty.
        """
        tf.debugging.Assert(dataset is not None, [])
        N = tf.shape(dataset.query_points)[0]

        quality_scores = self._quality_function(model, dataset)

        chosen_inducing_points = greedy_inference_dpp(
            M=tf.minimum(M, N),
            kernel=model.get_kernel(),
            quality_scores=quality_scores,
            dataset=dataset,
        )  # [min(M,N), d]

        if N < M:  # if fewer data than inducing points then sample remaining uniformly
            data_as_discrete_search_space = DiscreteSearchSpace(dataset.query_points)
            convex_hull_of_data = Box(
                lower=data_as_discrete_search_space.lower,
                upper=data_as_discrete_search_space.upper,
            )
            uniform_sampler = UniformInducingPointSelector(convex_hull_of_data)
            uniform_sample = uniform_sampler._recalculate_inducing_points(
                M - N,
                model,
                dataset,
            )  # [M-N, d]
            chosen_inducing_points = tf.concat(
                [chosen_inducing_points, uniform_sample], 0
            )  # [M, d]
        return chosen_inducing_points  # [M, d]


class UnitQualityFunction(QualityFunction):
    """
    A :class:`QualityFunction` where all points are considered equal, i.e. using
    this quality function for inducing point allocation corresponds to allocating
    inducing points with the sole aim of minimizing predictive variance.
    """

    def __call__(self, model: GPflowPredictor, dataset: Dataset) -> TensorType:
        """
        Evaluate the quality of the data-points according to the model.
        :param model: The sparse model.
        :param dataset: The data from the observer. Must be populated.
        :return: The quality scores.
        """

        return tf.ones(tf.shape(dataset.query_points)[0], dtype=tf.float64)  # [N]


class ModelBasedImprovementQualityFunction(QualityFunction):
    """
    A :class:`QualityFunction` where the quality of points are given by their expected
    improvement with respect to a conservative baseline. Expectations are according
    to the model from the previous BO step). See :cite:`moss2023IPA` for details
    and justification.
    """

    def __call__(self, model: GPflowPredictor, dataset: Dataset) -> TensorType:
        """
        Evaluate the quality of the data-points according to the model.
        :param model: The sparse model.
        :param dataset: The data from the observer. Must be populated.
        :return: The quality scores.
        """

        mean, variance = model.predict(dataset.query_points)  # [N, 1], [N, 1]
        baseline = tf.reduce_max(mean)
        normal = tfp.distributions.Normal(mean, tf.sqrt(variance))
        improvement = (baseline - mean) * normal.cdf(baseline) + variance * normal.prob(
            baseline
        )  # [N, 1]
        return improvement[:, 0]  # [N]


class ConditionalVarianceReduction(DPPInducingPointSelector):
    """
    An :class:`InducingPointSelector` that greedily chooses the points with maximal (conditional)
    predictive variance, see :cite:`burt2019rates`.
    """

    def __init__(self, recalc_every_model_update: bool = True):
        """
        :param recalc_every_model_update: If True then recalculate the inducing points for each
            model update, otherwise just recalculate on the first call.
        """

        super().__init__(UnitQualityFunction(), recalc_every_model_update)


class ConditionalImprovementReduction(DPPInducingPointSelector):
    """
    An :class:`InducingPointSelector` that greedily chooses points with large predictive variance
    and that are likely to be in promising regions of the search space, see :cite:`moss2023IPA`.
    """

    def __init__(
        self,
        recalc_every_model_update: bool = True,
    ):
        """
        :param recalc_every_model_update: If True then recalculate the inducing points for each
            model update, otherwise just recalculate on the first call.
        """

        super().__init__(ModelBasedImprovementQualityFunction(), recalc_every_model_update)


def greedy_inference_dpp(
    M: int,
    kernel: gpflow.kernels.Kernel,
    quality_scores: TensorType,
    dataset: Dataset,
) -> TensorType:
    """
    Get a greedy approximation of the MAP estimate of the Determinantal Point Process (DPP)
    over ``dataset`` following the algorithm of :cite:`chen2018fast`. Note that we are using the
    quality-diversity decomposition of a DPP, specifying both a similarity ``kernel``
    and ``quality_scores``.

    :param M: Desired set size.
    :param kernel: The underlying kernel of the DPP.
    :param quality_scores: The quality score of each item in ``dataset``.
    :return: The MAP estimate of the DPP.
    :raise tf.errors.InvalidArgumentError: If ``dataset`` is empty or if the shape of
        ``quality_scores`` does not match that of ``dataset.observations``.
    """
    tf.debugging.Assert(dataset is not None, [])
    tf.debugging.assert_equal(tf.shape(dataset.observations)[0], tf.shape(quality_scores)[0])
    tf.debugging.Assert(len(dataset.query_points) >= M, [])

    chosen_indicies = []  # iteratively store chosen points

    N = tf.shape(dataset.query_points)[0]
    c = tf.zeros((M - 1, N))  # [M-1,N]
    d_squared = kernel.K_diag(dataset.query_points)  # [N]

    scores = d_squared * quality_scores**2  # [N]
    chosen_indicies.append(tf.argmax(scores))  # get first element
    for m in range(M - 1):  # get remaining elements
        ix = tf.cast(chosen_indicies[-1], dtype=tf.int32)  # increment Cholesky with newest point
        newest_point = dataset.query_points[ix]

        d_temp = tf.math.sqrt(d_squared[ix])  # [1]

        L = kernel.K(dataset.query_points, newest_point[None, :])[:, 0]  # [N]
        if m == 0:
            e = L / d_temp
            c = tf.expand_dims(e, 0)  # [1,N]
        else:
            c_temp = c[:, ix : ix + 1]  # [m,1]
            e = (L - tf.matmul(tf.transpose(c_temp), c[:m])) / d_temp  # [N]
            c = tf.concat([c, e], axis=0)  # [m+1, N]
            e = tf.squeeze(e, 0)

        d_squared -= e**2
        d_squared = tf.maximum(d_squared, 1e-50)  # numerical stability

        scores = d_squared * quality_scores**2  # [N]
        chosen_indicies.append(tf.argmax(scores))  # get next element as point with largest score

    return tf.gather(dataset.query_points, chosen_indicies)
