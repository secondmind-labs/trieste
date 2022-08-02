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

import tensorflow as tf
import numpy as np
from cvxopt import matrix
from cvxopt.solvers import qp

from ...types import TensorType
from .dominance import non_dominated
from .partition import prepare_default_non_dominated_partition_bounds


class Pareto:
    """
    A :class:`Pareto` constructs a Pareto set.
    Stores a Pareto set and calculates hypervolume of the Pareto set given a
    specified reference point
    """

    def __init__(
        self,
        observations: TensorType,
        already_non_dominated: bool = False
    ):
        """
        :param observations: The observations for all objectives, with shape [N, D].
        :param already_non_dominated: Bool of whether the points are already non dominated
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

    def sample(self, sample_size: int):
        """
        Sample a set of diverse points from the Pareto set using
        Hypervolume Sharpe-Ratio Indicator
        """

        front_size, front_dims = self.front.shape

        if front_dims != 2:
            raise NotImplementedError("Pareto front sampling is only supported in the 2D case")
        # Define lower bound and reference point
        lower_bound = [float(min(self.front[:, i])) for i in range(front_dims)]

        # Calculate the deltas to add to the upper bound to get the reference point
        u_deltas = [(float(max(self.front[:, i])) - float(min(self.front[:, i]))) * 0.2 for i in range(front_dims)]

        # Use deltas and max values to create reference point
        reference_point = [float(max(self.front[:,i])) + u_deltas[i] for i in range(front_dims)]

        # Calculate p matrix
        p = np.zeros([front_size, front_size])

        # Calcualte denominator value for p matrix elements 
        denominator = 1
        for i in range(front_dims):
            denominator *= reference_point[i] - lower_bound[i]
        
        # Fill entries of p
        for i in range(front_size):
            for j in range(front_size):
                p[i,j] = ((reference_point[0] - max(self.front[i, 0],self.front[j, 0])) * ([1] - max(self.front[i, 1], self.front[j, 1])))

        p = p / denominator

        # Calculate q
        p_diag = np.expand_dims(np.diagonal(p), axis=1)
        q = p - np.dot(p_diag, np.transpose(p_diag))

        # Solve quadratic programming problem for y*
        P = matrix(np.array(q))
        q = matrix(np.zeros([front_size, 1]))
        G = matrix(-1*np.eye(front_size))
        h = matrix(np.zeros([front_size, 1]))
        A = matrix(np.transpose(p_diag))
        b = matrix(np.ones([1,1]))
        optim = qp(P=P,q=q,G=G,h=h,A=A,b=b)
        
        # Extract y*
        y_star = np.array(optim["x"])
        # Calculate x*
        x_star = y_star / np.sum(y_star)

        # Create id array to keep track of points
        id_arr = np.expand_dims(np.arange(front_size), axis=1)

        # Stitch id array, x_star and the front together
        stitched_array = np.concatenate([id_arr, x_star, np.array(self.front)], axis=1)

        # Sort array by x_star descending
        sorted_array = stitched_array[stitched_array[:,0].argsort()[::-1]]

        samples = sorted_array[:sample_size, 2:]
        sample_ids = sorted_array[:sample_size, 0].astype(int)

        return samples, sample_ids



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
