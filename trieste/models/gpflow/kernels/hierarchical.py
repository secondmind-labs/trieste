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
"""Arc and Wedge kernels for hierarchical / conditional search spaces.

These kernels handle conditional variables by embedding each conditional dimension
into R^2, so that the three axiomatic distance cases (both-inactive, both-active,
incomparable) are captured geometrically. Unconditional dimensions pass through
unchanged. A user-supplied base kernel evaluates covariance in the embedded space.

References:
    - Swersky et al. (2014) — Arc (cylindrical) kernel
    - Horn et al. (2019)    — Wedge (triangular) kernel
"""
from __future__ import annotations

from typing import Optional

import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.base import TensorLike
from gpflow.utilities import positive, to_default_float

from ....space import HierarchicalSearchSpace
from ....types import TensorType


_PI = tf.constant(np.pi, dtype=gpflow.default_float())


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _build_activity_mask(
    points: TensorType, space: HierarchicalSearchSpace
) -> TensorType:
    """Per-dimension boolean mask over non-indicator dimensions.

    For each non-indicator subspace, the corresponding columns are ``True``
    when the `HierarchyNode` that owns that subspace has its
    ``indicator_conditions`` satisfied by the indicator values in *points*.
    Unconditional nodes (empty ``indicator_conditions``) always return True.

    :param points: Flat input tensor of shape ``[N, D]``.
    :param space: The hierarchical search space.
    :return: Boolean mask of shape ``[N, n_non_indicator_dims]``.
    """
    masks = []
    for tag in space.non_indicator_tags:
        sub_dim = int(space.get_subspace(tag).dimension)
        nodes = space.node_for_subspace(tag)
        if not nodes:
            masks.append(tf.ones([tf.shape(points)[0], sub_dim], dtype=tf.bool))
            continue

        node = nodes[0]
        if not node.indicator_conditions:
            masks.append(tf.ones([tf.shape(points)[0], sub_dim], dtype=tf.bool))
            continue

        active = tf.ones([tf.shape(points)[0]], dtype=tf.bool)
        for ind_tag, required in node.indicator_conditions.items():
            ind_vals = space.get_subspace_component(ind_tag, points)
            ind_vals = tf.squeeze(ind_vals, axis=-1)
            if required:
                cond = tf.greater(ind_vals, 0.5)
            else:
                cond = tf.less_equal(ind_vals, 0.5)
            active = tf.logical_and(active, cond)

        active_col = tf.expand_dims(active, -1)
        masks.append(tf.repeat(active_col, sub_dim, axis=-1))

    return tf.concat(masks, axis=-1)


def _build_bounds_tensor(space: HierarchicalSearchSpace) -> TensorType:
    """Lower/upper bounds for every non-indicator dimension, in tag order.

    :param space: The hierarchical search space.
    :return: Tensor of shape ``[n_non_indicator_dims, 2]`` with ``(lower, upper)`` pairs.
    """
    lowers, uppers = [], []
    for tag in space.non_indicator_tags:
        sub = space.get_subspace(tag)
        lowers.append(tf.cast(sub.lower, gpflow.default_float()))
        uppers.append(tf.cast(sub.upper, gpflow.default_float()))
    lo = tf.concat(lowers, axis=0)
    hi = tf.concat(uppers, axis=0)
    return tf.stack([lo, hi], axis=-1)


def _extract_non_indicator_dims(
    points: TensorType, space: HierarchicalSearchSpace
) -> TensorType:
    """Extract and concatenate all non-indicator subspace columns.

    :param points: Flat input tensor of shape ``[N, D]``.
    :param space: The hierarchical search space.
    :return: Tensor of shape ``[N, n_non_indicator_dims]``.
    """
    parts = [space.get_subspace_component(tag, points) for tag in space.non_indicator_tags]
    return tf.concat(parts, axis=-1)


def _classify_dims(
    space: HierarchicalSearchSpace,
) -> tuple[list[int], list[int]]:
    """Classify non-indicator dimensions as unconditional or conditional.

    A dimension is *conditional* if its subspace's ``HierarchyNode`` has
    non-empty ``indicator_conditions``.

    :param space: The hierarchical search space.
    :return: ``(unconditional_indices, conditional_indices)`` as flat lists
        of integer positions relative to the non-indicator vector.
    """
    uncond: list[int] = []
    cond: list[int] = []
    offset = 0
    for tag in space.non_indicator_tags:
        sub_dim = int(space.get_subspace(tag).dimension)
        nodes = space.node_for_subspace(tag)
        is_conditional = any(bool(n.indicator_conditions) for n in nodes)
        indices = list(range(offset, offset + sub_dim))
        if is_conditional:
            cond.extend(indices)
        else:
            uncond.extend(indices)
        offset += sub_dim
    return uncond, cond


# ---------------------------------------------------------------------------
# ArcKernel
# ---------------------------------------------------------------------------


class ArcKernel(gpflow.kernels.Kernel):
    """Cylindrical arc kernel for hierarchical search spaces (Swersky et al., 2014).

    Each conditional dimension is mapped to R^2 via::

        active:   [radius * sin(pi * angle * v),  radius * cos(pi * angle * v)]
        inactive: [0, 0]

    where ``v = (x - lower) / (upper - lower)`` is the normalised value.
    Unconditional dimensions are normalised and passed through unchanged.
    A user-supplied ``base_kernel`` evaluates covariance in the embedded space
    (its lengthscale is frozen at 1.0).

    :param space: A :class:`HierarchicalSearchSpace` defining the hierarchy.
    :param base_kernel: GPflow kernel applied in the embedded space.
    :param angle_prior: Optional prior on the ``angle`` parameter.
    :param radius_prior: Optional prior on the ``radius`` parameter.
    """

    def __init__(
        self,
        space: HierarchicalSearchSpace,
        base_kernel: Optional[gpflow.kernels.Kernel] = None,
        angle_prior: Optional[TensorLike] = None,
        radius_prior: Optional[TensorLike] = None,
    ) -> None:
        super().__init__()
        self.space = space
        self._bounds = _build_bounds_tensor(space)
        self._uncond_idx, self._cond_idx = _classify_dims(space)
        self._n_cond = len(self._cond_idx)

        if base_kernel is None:
            base_kernel = gpflow.kernels.Matern52()
        if hasattr(base_kernel, "lengthscales"):
            base_kernel.lengthscales.assign(
                tf.ones_like(base_kernel.lengthscales)
            )
            gpflow.utilities.set_trainable(base_kernel.lengthscales, False)
        self.base_kernel = base_kernel

        if self._n_cond > 0:
            angle_init = 0.5 * tf.ones(self._n_cond, dtype=gpflow.default_float())
            self.angle = gpflow.Parameter(
                angle_init,
                transform=tfp.bijectors.Sigmoid(
                    to_default_float(0.1), to_default_float(0.9)
                ),
                prior=angle_prior,
                name="angle",
            )
            radius_init = tf.ones(self._n_cond, dtype=gpflow.default_float())
            self.radius = gpflow.Parameter(
                radius_init, transform=positive(), prior=radius_prior, name="radius"
            )

    def _embed(self, X: TensorType) -> TensorType:
        """Map flat inputs into the arc-embedded space.

        :param X: Input tensor of shape ``[N, D]``.
        :return: Embedded tensor of shape ``[N, n_uncond + 2 * n_cond]``.
        """
        X_ni = _extract_non_indicator_dims(X, self.space)
        X_ni = tf.cast(X_ni, gpflow.default_float())
        mask = _build_activity_mask(X, self.space)
        mask_f = tf.cast(mask, gpflow.default_float())

        lo = self._bounds[:, 0]
        hi = self._bounds[:, 1]
        ranges = hi - lo
        ranges = tf.where(tf.abs(ranges) < 1e-12, tf.ones_like(ranges), ranges)
        v = (X_ni - lo) / ranges

        parts = []

        if self._uncond_idx:
            uncond_v = tf.gather(v, self._uncond_idx, axis=-1)
            parts.append(uncond_v)

        if self._n_cond > 0:
            cond_v = tf.gather(v, self._cond_idx, axis=-1)
            cond_mask = tf.gather(mask_f, self._cond_idx, axis=-1)

            theta = _PI * self.angle * cond_v
            sin_part = self.radius * tf.sin(theta) * cond_mask
            cos_part = self.radius * tf.cos(theta) * cond_mask
            parts.extend([sin_part, cos_part])

        return tf.concat(parts, axis=-1)

    def K(self, X: TensorType, X2: Optional[TensorType] = None) -> TensorType:
        Z1 = self._embed(X)
        Z2 = self._embed(X2) if X2 is not None else Z1
        return self.base_kernel.K(Z1, Z2)

    def K_diag(self, X: TensorType) -> TensorType:
        return self.base_kernel.K_diag(self._embed(X))


# ---------------------------------------------------------------------------
# WedgeKernel
# ---------------------------------------------------------------------------


class WedgeKernel(gpflow.kernels.Kernel):
    """Triangular wedge kernel for hierarchical search spaces (Horn et al., 2019).

    Each conditional dimension is mapped to R^2 via::

        active:   [theta1 * v + theta2 * v * cos(rho),  theta2 * v * sin(rho)]
        inactive: [0, 0]

    where ``v = (x - lower) / (upper - lower)`` is the normalised value.
    Unlike the arc kernel, the incomparable distance (one active, one inactive)
    depends on the active-side value, which is strictly more informative near
    disjunction boundaries.

    :param space: A :class:`HierarchicalSearchSpace` defining the hierarchy.
    :param base_kernel: GPflow kernel applied in the embedded space.
    :param theta1_prior: Optional prior on the ``theta1`` parameter.
    :param theta2_prior: Optional prior on the ``theta2`` parameter.
    :param rho_prior: Optional prior on the ``rho`` parameter.
    """

    def __init__(
        self,
        space: HierarchicalSearchSpace,
        base_kernel: Optional[gpflow.kernels.Kernel] = None,
        theta1_prior: Optional[TensorLike] = None,
        theta2_prior: Optional[TensorLike] = None,
        rho_prior: Optional[TensorLike] = None,
    ) -> None:
        super().__init__()
        self.space = space
        self._bounds = _build_bounds_tensor(space)
        self._uncond_idx, self._cond_idx = _classify_dims(space)
        self._n_cond = len(self._cond_idx)

        if base_kernel is None:
            base_kernel = gpflow.kernels.Matern52()
        if hasattr(base_kernel, "lengthscales"):
            base_kernel.lengthscales.assign(
                tf.ones_like(base_kernel.lengthscales)
            )
            gpflow.utilities.set_trainable(base_kernel.lengthscales, False)
        self.base_kernel = base_kernel

        if self._n_cond > 0:
            theta1_init = tf.ones(self._n_cond, dtype=gpflow.default_float())
            self.theta1 = gpflow.Parameter(
                theta1_init, transform=positive(), prior=theta1_prior, name="theta1"
            )

            theta2_init = tf.ones(self._n_cond, dtype=gpflow.default_float())
            self.theta2 = gpflow.Parameter(
                theta2_init, transform=positive(), prior=theta2_prior, name="theta2"
            )

            rho_init = 0.5 * _PI * tf.ones(self._n_cond, dtype=gpflow.default_float())
            self.rho = gpflow.Parameter(
                rho_init,
                transform=tfp.bijectors.Sigmoid(
                    to_default_float(1e-6), to_default_float(np.pi)
                ),
                prior=rho_prior,
                name="rho",
            )

    def _embed(self, X: TensorType) -> TensorType:
        """Map flat inputs into the wedge-embedded space.

        :param X: Input tensor of shape ``[N, D]``.
        :return: Embedded tensor of shape ``[N, n_uncond + 2 * n_cond]``.
        """
        X_ni = _extract_non_indicator_dims(X, self.space)
        X_ni = tf.cast(X_ni, gpflow.default_float())
        mask = _build_activity_mask(X, self.space)
        mask_f = tf.cast(mask, gpflow.default_float())

        lo = self._bounds[:, 0]
        hi = self._bounds[:, 1]
        ranges = hi - lo
        ranges = tf.where(tf.abs(ranges) < 1e-12, tf.ones_like(ranges), ranges)
        v = (X_ni - lo) / ranges

        parts = []

        if self._uncond_idx:
            uncond_v = tf.gather(v, self._uncond_idx, axis=-1)
            parts.append(uncond_v)

        if self._n_cond > 0:
            cond_v = tf.gather(v, self._cond_idx, axis=-1)
            cond_mask = tf.gather(mask_f, self._cond_idx, axis=-1)

            comp1 = (self.theta1 * cond_v + self.theta2 * cond_v * tf.cos(self.rho)) * cond_mask
            comp2 = (self.theta2 * cond_v * tf.sin(self.rho)) * cond_mask
            parts.extend([comp1, comp2])

        return tf.concat(parts, axis=-1)

    def K(self, X: TensorType, X2: Optional[TensorType] = None) -> TensorType:
        Z1 = self._embed(X)
        Z2 = self._embed(X2) if X2 is not None else Z1
        return self.base_kernel.K(Z1, Z2)

    def K_diag(self, X: TensorType) -> TensorType:
        return self.base_kernel.K_diag(self._embed(X))
