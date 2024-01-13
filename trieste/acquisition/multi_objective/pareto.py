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
""" This module contains functions and classes for Pareto based multi-objective optimization. """
from __future__ import annotations

try:
    import cvxpy as cp
except ImportError:  # pragma: no cover (tested but not by coverage)
    cp = None
import numpy as np
import tensorflow as tf

from ...types import TensorType
from .dominance import non_dominated
from .partition import prepare_default_non_dominated_partition_bounds


class Pareto:
    """
    A :class:`Pareto` constructs a Pareto set.
    Stores a Pareto set and calculates hypervolume of the Pareto set given a
    specified reference point
    """

    def __init__(self, observations: TensorType, already_non_dominated: bool = False):
        """
        :param observations: The observations for all objectives, with shape [N, D].
        :param already_non_dominated: Whether the observations are already non dominated
        :raise ValueError (or InvalidArgumentError): If ``observations`` has an invalid shape.
        """
        tf.debugging.assert_rank(observations, 2)
        tf.debugging.assert_greater_equal(tf.shape(observations)[-1], 2)

        if not already_non_dominated:
            self.front = non_dominated(observations)[0]
        else:
            self.front = observations

    def hypervolume_indicator(self, reference: TensorType) -> TensorType:
        """
        Calculate the hypervolume indicator based on self.front and a reference point
        The hypervolume indicator is the volume of the dominated region.

        :param reference: a reference point to use, with shape [D].
            Defines the upper bound of the hypervolume.
            Should be equal or bigger than the anti-ideal point of the Pareto set.
            For comparing results across runs, the same reference point must be used.
        :return: hypervolume indicator, if reference point is less than all of the front
            in any dimension, the hypervolume indicator will be zero.
        :raise ValueError (or `tf.errors.InvalidArgumentError`): If ``reference`` has an invalid
            shape.
        :raise ValueError (or `tf.errors.InvalidArgumentError`): If ``self.front`` is empty
            (which can happen if the concentration point is too strict so no frontier
            exists after the screening)
        """
        if tf.equal(tf.size(self.front), 0):
            raise ValueError("empty front cannot be used to calculate hypervolume indicator")

        helper_anti_reference = tf.reduce_min(self.front, axis=0) - tf.ones(
            shape=1, dtype=self.front.dtype
        )
        lower, upper = prepare_default_non_dominated_partition_bounds(
            reference, self.front, helper_anti_reference
        )
        non_dominated_hypervolume = tf.reduce_sum(tf.reduce_prod(upper - lower, 1))
        hypervolume_indicator = (
            tf.reduce_prod(reference - helper_anti_reference) - non_dominated_hypervolume
        )
        return hypervolume_indicator

    def sample_diverse_subset(
        self,
        sample_size: int,
        allow_repeats: bool = True,
        bounds_delta_scale_factor: float = 0.2,
        bounds_min_delta: float = 1e-9,
    ) -> tuple[TensorType, TensorType]:
        """
        Sample a set of diverse points from the Pareto set using
        Hypervolume Sharpe-Ratio Indicator

        :param sample_size: The number of points to sample from the Pareto front
        :param allow_repeats: Whether the sample may contain repeats
        :param bounds_delta_scale_factor: The factor by which to grow the distance
            between extrema when calculating lower and upper bounds
        :param bounds_min_delta: The minimum value of the distance between extrema
        :return: sample: Tensor of query points selected in the sample
                 and sample_ids: Tensor of indices of points selected from the Pareto set
        """

        if cp is None:
            raise ImportError(
                "Pareto.sample method requires cvxpy, "
                "this can be installed via `pip install trieste[qhsri]`"
            )

        if bounds_delta_scale_factor < 0:
            raise ValueError(
                "bounds_delta_scale_factor should be non-negative,"
                f" got {bounds_delta_scale_factor}"
            )
        if bounds_delta_scale_factor < 0:
            raise ValueError("bounds_delta_min should be non-negative, got {bounds_min_delta}")

        front_size, _ = self.front.shape

        if (front_size < sample_size) and allow_repeats is False:
            raise ValueError(
                f"Tried to sample {sample_size} points from a Pareto"
                f" set of size {front_size}, please ensure sample size is smaller than"
                " Pareto set size."
            )

        lower_bound, upper_bound = self._get_bounds(bounds_delta_scale_factor, bounds_min_delta)

        p = self._calculate_p_matrix(lower_bound, upper_bound)

        # Calculate q matrix
        p_diag = np.expand_dims(np.diagonal(p), axis=1)
        q = p - np.dot(p_diag, np.transpose(p_diag))

        x_star = self._find_x_star(q, p)

        if allow_repeats:
            samples, sample_ids = self._choose_batch_with_repeats(x_star, sample_size)
        else:
            samples, sample_ids = self._choose_batch_no_repeats(x_star, sample_size)

        return samples, sample_ids

    def _choose_batch_with_repeats(
        self, x_star: TensorType, sample_size: int
    ) -> tuple[TensorType, TensorType]:
        # Calculate number of times each point is sampled
        n_times_sampled = x_star * sample_size

        # Round each number to an int
        n_times_sampled = np.round(n_times_sampled)

        # Check difference in number of samples and required sample size
        n_samples_difference = sample_size - int(np.sum(n_times_sampled))

        if n_samples_difference < 0:
            # We need to randomly remove samples
            # Get indices of point that were to be sampled >0 times
            non_zero_indices = np.flatnonzero(n_times_sampled)
            # Choose indices to decrement
            samples_to_decr = np.random.choice(non_zero_indices, size=-n_samples_difference)
            # Decrement indices
            for idx in samples_to_decr:
                n_times_sampled[idx] -= 1
        elif n_samples_difference > 0:
            # We need to randomly add samples
            samples_to_incr = np.random.choice(
                np.arange(len(n_times_sampled)), size=n_samples_difference
            )
            for idx in samples_to_incr:
                n_times_sampled[idx] += 1

        # Create a list of the sample indices
        sample_ids = []
        for idx, repeats in enumerate(n_times_sampled):
            for _ in range(int(repeats)):
                sample_ids.append(idx)
        # Convert to array for indexing and return
        sample_ids_array = np.array(sample_ids)

        # Create batch with each of the selected samples
        samples = np.array(self.front)[sample_ids_array]

        return samples, sample_ids_array

    def _choose_batch_no_repeats(
        self, x_star: TensorType, sample_size: int
    ) -> tuple[TensorType, TensorType]:
        front_size = self.front.shape[0]

        # Create id array to keep track of points
        id_arr = np.expand_dims(np.arange(front_size), axis=1)

        # Stitch id array, x_star and the front together
        stitched_array = np.concatenate([id_arr, x_star, np.array(self.front)], axis=1)

        # Sort array by x_star descending
        sorted_array = stitched_array[stitched_array[:, 1].argsort()[::-1]]

        samples = sorted_array[:sample_size, 2:]
        sample_ids = sorted_array[:sample_size, 0].astype(int)

        return samples, sample_ids

    def _find_x_star(self, q: TensorType, p: TensorType) -> TensorType:
        front_size = self.front.shape[0]

        p_diag = np.expand_dims(np.diagonal(p), axis=1)

        # Solve quadratic program for y*
        P = cp.atoms.affine.wraps.psd_wrap(q)
        G = np.eye(front_size)
        h = np.zeros(front_size)
        A = np.transpose(p_diag)
        b = np.ones(1)

        # Define and solve the CVXPY problem.
        y = cp.Variable(front_size)
        prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(y, P)), [G @ y >= h, A @ y == b])
        prob.solve()

        y_star = y.value

        # Calculate x*
        x_star = np.expand_dims(y_star, axis=1) / np.sum(y_star)

        return x_star

    def _calculate_p_matrix(self, lower_bound: TensorType, upper_bound: TensorType) -> TensorType:
        front_size, front_dims = self.front.shape

        p = np.zeros([front_size, front_size])

        # Calculate denominator value for p matrix elements
        denominator: float = 1
        for i in range(front_dims):
            if upper_bound[i] - lower_bound[i] == 0:
                raise ValueError(
                    "Pareto set has identical upper and lower bounds"
                    " in a dimension, you can avoid this by setting a "
                    "nonzero value for bounds_min_delta"
                )
            denominator *= upper_bound[i] - lower_bound[i]

        # Fill entries of p
        for i in range(front_size):
            for j in range(front_size):
                pij = 1
                for k in range(front_dims):
                    pij *= upper_bound[k] - max(self.front[i, k], self.front[j, k])
                p[i, j] = pij

        p = p / denominator

        return p

    def _get_bounds(
        self, delta_scaling_factor: float, min_delta: float
    ) -> tuple[TensorType, TensorType]:
        # Find min and max for each dimension in the front
        dim_mins = np.min(self.front, axis=0)
        dim_maxes = np.max(self.front, axis=0)
        # Calculate the deltas to add to the min/max to get the upper and lower bounds
        deltas = ((dim_maxes - dim_mins) * delta_scaling_factor) + min_delta
        # Calculate the bounds
        lower_bound = dim_mins - deltas
        upper_bound = dim_maxes + deltas

        return lower_bound, upper_bound


def get_reference_point(
    observations: TensorType,
) -> TensorType:
    """
    Default reference point calculation method that calculates the reference
    point according to a Pareto front extracted from set of observations.

    :param observations: observations referred to calculate the reference
        point, with shape [..., N, D]
    :return: a reference point to use, with shape [..., D].
    :raise ValueError: If ``observations`` is empty
    """
    if tf.equal(tf.size(observations), 0):
        raise ValueError("empty observations cannot be used to calculate reference point")

    front = Pareto(observations).front
    f = tf.math.reduce_max(front, axis=-2) - tf.math.reduce_min(front, axis=-2)
    return tf.math.reduce_max(front, axis=-2) + 2 * f / tf.cast(tf.shape(front)[-2], f.dtype)
