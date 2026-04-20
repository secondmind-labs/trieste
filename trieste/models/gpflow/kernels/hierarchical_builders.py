# Copyright 2024 The Trieste Contributors
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
"""Trieste-side factories that construct :class:`ArcKernel` / :class:`WedgeKernel`
from a :class:`HierarchicalSearchSpace`.

The kernels themselves live in :mod:`trieste.models.gpflow.kernels.hierarchical`
and depend only on GPflow/TF primitives. This module is the single place where
the hierarchy description (tags, nodes, indicator conditions, per-subspace
bounds) is translated into the integer-index primitives the kernels expect,
so the kernel module can later be migrated into GPflow without changes.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import gpflow
import tensorflow as tf
import tensorflow_probability as tfp

from ....space import HierarchicalSearchSpace
from .hierarchical import ActivityCondition, ArcKernel, WedgeKernel


def primitives_from_space(
    space: HierarchicalSearchSpace,
) -> Tuple[List[int], tf.Tensor, List[int], List[ActivityCondition]]:
    """Extract the pure-gpflow kernel primitives from a hierarchical search space.

    :param space: The hierarchical search space describing indicator tags,
        non-indicator subspaces, and the ``HierarchyNode`` activity rules.
    :return: ``(feature_dims, feature_bounds, indicator_dims, activity_conditions)``
        where

        - ``feature_dims`` lists the flat-vector column indices of each
          non-indicator feature (with multi-dimensional subspaces expanded
          into one entry per dim), in ``space.non_indicator_tags`` order;
        - ``feature_bounds`` is a ``[D_f, 2]`` ``float64`` tensor of
          ``(lower, upper)`` pairs for each feature;
        - ``indicator_dims`` lists the flat-vector column indices of the
          indicator subspaces in ``space.indicator_tags`` order;
        - ``activity_conditions`` contains one :class:`ActivityCondition`
          per feature column (in ``feature_dims`` order), with
          ``feature_dim`` set to that column and ``requirements`` populated
          from the owning :class:`HierarchyNode`'s ``indicator_conditions``
          (translating indicator tags into local indices into
          ``indicator_dims``). Unconditional features produce an
          :class:`ActivityCondition` with empty requirements.
    """
    indicator_tags = list(space.indicator_tags)
    indicator_local_by_tag = {tag: k for k, tag in enumerate(indicator_tags)}

    indicator_dims: List[int] = []
    for tag in indicator_tags:
        start = int(space._subspace_starting_indices[tag])
        size = int(space._subspace_sizes_by_tag[tag])
        if size != 1:
            raise ValueError(
                f"Indicator subspace '{tag}' must be one-dimensional; got size {size}."
            )
        indicator_dims.append(start)

    feature_dims: List[int] = []
    lowers: List[tf.Tensor] = []
    uppers: List[tf.Tensor] = []
    activity_conditions: List[ActivityCondition] = []

    for tag in space.non_indicator_tags:
        sub = space.get_subspace(tag)
        sub_dim = int(sub.dimension)
        start = int(space._subspace_starting_indices[tag])
        feature_dims.extend(range(start, start + sub_dim))
        lowers.append(tf.cast(sub.lower, gpflow.default_float()))
        uppers.append(tf.cast(sub.upper, gpflow.default_float()))

        nodes = space.node_for_subspace(tag)
        if nodes and nodes[0].indicator_conditions:
            requirements = {
                indicator_local_by_tag[ind_tag]: bool(required)
                for ind_tag, required in nodes[0].indicator_conditions.items()
            }
        else:
            requirements = {}

        for c in range(start, start + sub_dim):
            activity_conditions.append(
                ActivityCondition(feature_dim=c, requirements=dict(requirements))
            )

    if lowers:
        lo = tf.concat(lowers, axis=0)
        hi = tf.concat(uppers, axis=0)
        feature_bounds = tf.stack([lo, hi], axis=-1)
    else:
        feature_bounds = tf.zeros([0, 2], dtype=gpflow.default_float())

    return feature_dims, feature_bounds, indicator_dims, activity_conditions


def arc_kernel_from_space(
    space: HierarchicalSearchSpace,
    base_kernel: Optional[gpflow.kernels.Kernel] = None,
    angle_prior: Optional[tfp.distributions.Distribution] = None,
    radius_prior: Optional[tfp.distributions.Distribution] = None,
) -> ArcKernel:
    """Construct an :class:`ArcKernel` from a :class:`HierarchicalSearchSpace`.

    :param space: The hierarchical search space.
    :param base_kernel: GPflow kernel applied in the embedded space.
    :param angle_prior: Optional prior on the ``angle`` parameter.
    :param radius_prior: Optional prior on the ``radius`` parameter.
    :return: An :class:`ArcKernel` configured to operate on points from
        ``space``.
    """
    feature_dims, feature_bounds, indicator_dims, activity_conditions = (
        primitives_from_space(space)
    )
    return ArcKernel(
        feature_dims=feature_dims,
        feature_bounds=feature_bounds,
        indicator_dims=indicator_dims,
        activity_conditions=activity_conditions,
        base_kernel=base_kernel,
        angle_prior=angle_prior,
        radius_prior=radius_prior,
    )


def wedge_kernel_from_space(
    space: HierarchicalSearchSpace,
    base_kernel: Optional[gpflow.kernels.Kernel] = None,
    theta1_prior: Optional[tfp.distributions.Distribution] = None,
    theta2_prior: Optional[tfp.distributions.Distribution] = None,
    rho_prior: Optional[tfp.distributions.Distribution] = None,
) -> WedgeKernel:
    """Construct a :class:`WedgeKernel` from a :class:`HierarchicalSearchSpace`.

    :param space: The hierarchical search space.
    :param base_kernel: GPflow kernel applied in the embedded space.
    :param theta1_prior: Optional prior on the ``theta1`` parameter.
    :param theta2_prior: Optional prior on the ``theta2`` parameter.
    :param rho_prior: Optional prior on the ``rho`` parameter.
    :return: A :class:`WedgeKernel` configured to operate on points from
        ``space``.
    """
    feature_dims, feature_bounds, indicator_dims, activity_conditions = (
        primitives_from_space(space)
    )
    return WedgeKernel(
        feature_dims=feature_dims,
        feature_bounds=feature_bounds,
        indicator_dims=indicator_dims,
        activity_conditions=activity_conditions,
        base_kernel=base_kernel,
        theta1_prior=theta1_prior,
        theta2_prior=theta2_prior,
        rho_prior=rho_prior,
    )
