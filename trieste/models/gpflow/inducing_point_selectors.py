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

import tensorflow as tf
from scipy.cluster.vq import kmeans

from ...data import Dataset
from ...space import Box, SearchSpace
from ...types import TensorType
from ..interfaces import ProbabilisticModel, ProbabilisticModelType


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

    def __init__(self, search_space: SearchSpace, recalc_every_model_update: bool = True):
        """
        :param search_space: The global search space over which the optimization is defined.
        :param recalc_every_model_update: If True then recalculate the inducing points for each
            model update, otherwise just recalculate on the first call.
        """

        self._search_space = search_space
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
        tf.debugging.Assert(current_inducing_points is not None, [])

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


class UniformInducingPointSelector(InducingPointSelector[ProbabilisticModel]):
    """
    An :class:`InducingPointSelector` that chooses points sampled uniformly across the search space.
    """

    def _recalculate_inducing_points(
        self, M: int, model: ProbabilisticModel, dataset: Dataset
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


class RandomSubSampleInducingPointSelector(InducingPointSelector[ProbabilisticModel]):
    """
    An :class:`InducingPointSelector` that chooses points at random from the training data.
    """

    def _recalculate_inducing_points(
        self, M: int, model: ProbabilisticModel, dataset: Dataset
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

        tf.debugging.Assert(len(dataset.query_points) is not None, [])

        N = tf.shape(dataset.query_points)[0]  # training data size
        shuffled_query_points = tf.random.shuffle(dataset.query_points)  # [N, d]
        sub_sample = shuffled_query_points[: tf.math.minimum(N, M), :]

        if N < M:  # if fewer data than inducing points then sample remaining uniformly
            uniform_sampler = UniformInducingPointSelector(self._search_space)
            uniform_sample = uniform_sampler._recalculate_inducing_points(
                M - N, model, dataset
            )  # [M-N, d]
            sub_sample = tf.concat([sub_sample, uniform_sample], 0)  # [M, d]
        return sub_sample  # [M, d]


class KMeansInducingPointSelector(InducingPointSelector[ProbabilisticModel]):
    """
    An :class:`InducingPointSelector` that chooses points as centroids of a K-means clustering
    of the training data.
    """

    def _recalculate_inducing_points(
        self, M: int, model: ProbabilisticModel, dataset: Dataset
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

        tf.debugging.Assert(dataset is not None, [])

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
            uniform_sampler = UniformInducingPointSelector(self._search_space)
            extra_centroids = uniform_sampler._recalculate_inducing_points(  # [M-C, d]
                M - len(centroids), model, dataset
            )
            centroids = tf.concat([centroids, extra_centroids], axis=0)  # [M, d]

        return centroids  # [M, d]
