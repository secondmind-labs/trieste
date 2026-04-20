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

The module is deliberately self-contained: it imports only from ``gpflow``,
``tensorflow``, ``tensorflow_probability``, and ``numpy``. The hierarchy is
described by pure primitives (integer column indices, a bounds tensor, and a
list of AND-conjunction activity conditions over indicator columns), so the
kernels can be migrated wholesale into GPflow in a later change.

References:
    - Swersky et al. (2014) -- Arc (cylindrical) kernel
    - Horn et al. (2019)    -- Wedge (triangular) kernel
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.base import TensorType
from gpflow.utilities import positive, to_default_float


_PI = tf.constant(np.pi, dtype=gpflow.default_float())

# Sentinel used in the compact ``required`` tensor to mark an indicator column
# that is irrelevant (ignored) for a particular feature's activity condition.
_IGNORE = -1


# ---------------------------------------------------------------------------
# Shared primitive helpers
# ---------------------------------------------------------------------------


def _compile_activity_conditions(
    activity_conditions: Sequence[Sequence[Tuple[int, bool]]],
    n_features: int,
    n_indicators: int,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Compile the per-feature AND-conjunction conditions into dense tensors.

    :param activity_conditions: For each feature ``j``, a sequence of
        ``(k, required_bool)`` pairs. ``k`` indexes into the indicator columns
        (``0 <= k < n_indicators``); ``required_bool`` is the value the
        indicator must take for feature ``j`` to be active. An empty sequence
        means the feature is unconditional (always active).
    :param n_features: Number of feature (non-indicator) dimensions ``D_f``.
    :param n_indicators: Number of indicator dimensions ``D_i``.
    :return: A tuple ``(required, is_ignore)`` where ``required`` is an
        ``int32`` tensor of shape ``[D_f, D_i]`` with entries in
        ``{-1, 0, 1}`` and ``is_ignore`` is a ``bool`` tensor of the same
        shape, True where ``required == -1``.
    """
    if len(activity_conditions) != n_features:
        raise ValueError(
            f"activity_conditions has length {len(activity_conditions)}, expected "
            f"{n_features} (one entry per feature dimension)."
        )

    required = np.full((n_features, n_indicators), _IGNORE, dtype=np.int32)
    for j, conds in enumerate(activity_conditions):
        for pair in conds:
            if len(pair) != 2:
                raise ValueError(
                    f"activity_conditions[{j}] entry {pair!r} must be a "
                    f"(indicator_index, required_bool) pair."
                )
            k, required_bool = pair
            if not (0 <= k < n_indicators):
                raise ValueError(
                    f"activity_conditions[{j}] references indicator index {k}; "
                    f"must be in [0, {n_indicators})."
                )
            val = 1 if bool(required_bool) else 0
            existing = required[j, k]
            if existing != _IGNORE and existing != val:
                raise ValueError(
                    f"activity_conditions[{j}] contains contradictory requirements "
                    f"for indicator {k}: both {bool(existing)} and {bool(required_bool)}."
                )
            required[j, k] = val

    required_t = tf.constant(required, dtype=tf.int32)
    is_ignore_t = tf.equal(required_t, _IGNORE)
    return required_t, is_ignore_t


def _classify_conditional(
    activity_conditions: Sequence[Sequence[Tuple[int, bool]]],
) -> Tuple[list[int], list[int]]:
    """Split feature dimensions into unconditional and conditional local indices.

    A feature dimension is *conditional* iff its entry in ``activity_conditions``
    is non-empty; otherwise it is unconditional.

    :param activity_conditions: Per-feature AND-conjunction conditions.
    :return: A tuple ``(unconditional_local_idx, conditional_local_idx)``
        listing positions within the feature-vector.
    """
    uncond: list[int] = []
    cond: list[int] = []
    for j, conds in enumerate(activity_conditions):
        if conds:
            cond.append(j)
        else:
            uncond.append(j)
    return uncond, cond


# ---------------------------------------------------------------------------
# Base class: shared embedding-then-base-kernel infrastructure
# ---------------------------------------------------------------------------


class _HierarchicalEmbeddingKernel(gpflow.kernels.Kernel):
    """Shared machinery for embedding-based hierarchical kernels.

    Subclasses override :meth:`_embed_conditional` to supply the per-dimension
    R^2 embedding for conditional features; the base class handles mask
    construction, unconditional pass-through, and delegation to the user's
    base kernel.
    """

    def __init__(
        self,
        feature_dims: Sequence[int],
        feature_bounds: TensorType,
        indicator_dims: Sequence[int] = (),
        activity_conditions: Sequence[Sequence[Tuple[int, bool]]] = (),
        base_kernel: Optional[gpflow.kernels.Kernel] = None,
    ) -> None:
        super().__init__()

        feature_dims = list(feature_dims)
        indicator_dims = list(indicator_dims)
        n_feat = len(feature_dims)
        n_ind = len(indicator_dims)

        bounds_tensor = tf.convert_to_tensor(feature_bounds, dtype=gpflow.default_float())
        if bounds_tensor.shape.rank != 2 or bounds_tensor.shape[-1] != 2:
            raise ValueError(
                f"feature_bounds must have shape [D_f, 2], got {bounds_tensor.shape}."
            )
        if int(bounds_tensor.shape[0]) != n_feat:
            raise ValueError(
                f"feature_bounds has {int(bounds_tensor.shape[0])} rows but "
                f"feature_dims has {n_feat} entries."
            )

        if not activity_conditions and n_feat > 0:
            activity_conditions = [() for _ in range(n_feat)]

        self._feature_dims = tf.constant(feature_dims, dtype=tf.int32)
        self._indicator_dims = tf.constant(indicator_dims, dtype=tf.int32)
        self._bounds = bounds_tensor
        self._n_feat = n_feat
        self._n_ind = n_ind

        self._required, self._required_is_ignore = _compile_activity_conditions(
            activity_conditions, n_feat, n_ind
        )

        uncond_local, cond_local = _classify_conditional(activity_conditions)
        self._uncond_local_idx = uncond_local
        self._cond_local_idx = cond_local
        self._n_uncond = len(uncond_local)
        self._n_cond = len(cond_local)

        if base_kernel is None:
            base_kernel = gpflow.kernels.Matern52()
        if hasattr(base_kernel, "lengthscales"):
            base_kernel.lengthscales.assign(tf.ones_like(base_kernel.lengthscales))
            gpflow.utilities.set_trainable(base_kernel.lengthscales, False)
        self.base_kernel = base_kernel

    # -- mask / embedding --------------------------------------------------

    def _build_activity_mask(self, X: TensorType) -> tf.Tensor:
        """Compute the per-feature boolean activity mask for a batch of points.

        :param X: Flat input tensor of shape ``[N, D]``.
        :return: Boolean tensor of shape ``[N, D_f]``; True means the
            corresponding feature dimension is active for that point.
        """
        if self._n_feat == 0:
            return tf.zeros([tf.shape(X)[0], 0], dtype=tf.bool)

        if self._n_ind == 0:
            return tf.ones([tf.shape(X)[0], self._n_feat], dtype=tf.bool)

        ind_vals = tf.gather(X, self._indicator_dims, axis=-1)
        ind_vals = tf.cast(ind_vals, gpflow.default_float())
        ind_int = tf.cast(ind_vals > to_default_float(0.5), tf.int32)  # [N, D_i]

        match = tf.logical_or(
            self._required_is_ignore[None, :, :],
            tf.equal(ind_int[:, None, :], self._required[None, :, :]),
        )  # [N, D_f, D_i]
        return tf.reduce_all(match, axis=-1)  # [N, D_f]

    def _embed(self, X: TensorType) -> tf.Tensor:
        """Map flat inputs into the embedded covariance space.

        :param X: Flat input tensor of shape ``[N, D]``.
        :return: Embedded tensor of shape ``[N, n_uncond + 2 * n_cond]``.
        """
        X = tf.cast(X, gpflow.default_float())
        X_feat = tf.gather(X, self._feature_dims, axis=-1)  # [N, D_f]
        mask_f = tf.cast(self._build_activity_mask(X), gpflow.default_float())

        lo = self._bounds[:, 0]
        hi = self._bounds[:, 1]
        ranges = hi - lo
        ranges = tf.where(tf.abs(ranges) < 1e-12, tf.ones_like(ranges), ranges)
        v = (X_feat - lo) / ranges  # [N, D_f]

        parts: list[tf.Tensor] = []

        if self._n_uncond > 0:
            parts.append(tf.gather(v, self._uncond_local_idx, axis=-1))

        if self._n_cond > 0:
            cond_v = tf.gather(v, self._cond_local_idx, axis=-1)
            cond_mask = tf.gather(mask_f, self._cond_local_idx, axis=-1)
            parts.extend(self._embed_conditional(cond_v, cond_mask))

        if not parts:
            return tf.zeros([tf.shape(X)[0], 0], dtype=gpflow.default_float())
        return tf.concat(parts, axis=-1)

    def _embed_conditional(
        self, cond_v: tf.Tensor, cond_mask: tf.Tensor
    ) -> list[tf.Tensor]:
        """Return the two ``[N, n_cond]`` tensors making up the R^2 embedding.

        Subclasses override this. The returned list is concatenated onto the
        unconditional features to form the embedded vector.
        """
        raise NotImplementedError

    # -- kernel interface --------------------------------------------------

    def K(self, X: TensorType, X2: Optional[TensorType] = None) -> tf.Tensor:
        Z1 = self._embed(X)
        Z2 = self._embed(X2) if X2 is not None else Z1
        return self.base_kernel.K(Z1, Z2)

    def K_diag(self, X: TensorType) -> tf.Tensor:
        return self.base_kernel.K_diag(self._embed(X))


# ---------------------------------------------------------------------------
# ArcKernel
# ---------------------------------------------------------------------------


class ArcKernel(_HierarchicalEmbeddingKernel):
    """Cylindrical arc kernel for hierarchical search spaces (Swersky et al., 2014).

    Each conditional feature dimension is mapped to R^2 via::

        active:   [radius * sin(pi * angle * v),  radius * cos(pi * angle * v)]
        inactive: [0, 0]

    where ``v = (x - lower) / (upper - lower)`` is the normalised value.
    Unconditional dimensions are normalised and passed through unchanged.
    A user-supplied ``base_kernel`` evaluates covariance in the embedded
    space (its lengthscale is frozen at 1.0).

    :param feature_dims: Indices into the flat input ``X`` giving the
        real-valued (non-indicator) feature columns, in the order matching
        ``feature_bounds`` and ``activity_conditions``.
    :param feature_bounds: ``[D_f, 2]`` tensor of ``(lower, upper)`` pairs
        for each feature dimension.
    :param indicator_dims: Indices into ``X`` of the 0/1 indicator columns
        used by ``activity_conditions``.
    :param activity_conditions: One entry per feature dimension giving a
        sequence of ``(indicator_local_index, required_bool)`` pairs that
        must all hold for the feature to be active. An empty sequence means
        the feature is unconditional.
    :param base_kernel: GPflow kernel applied in the embedded space.
    :param angle_prior: Optional prior on the ``angle`` parameter.
    :param radius_prior: Optional prior on the ``radius`` parameter.
    """

    def __init__(
        self,
        feature_dims: Sequence[int],
        feature_bounds: TensorType,
        indicator_dims: Sequence[int] = (),
        activity_conditions: Sequence[Sequence[Tuple[int, bool]]] = (),
        base_kernel: Optional[gpflow.kernels.Kernel] = None,
        angle_prior: Optional[tfp.distributions.Distribution] = None,
        radius_prior: Optional[tfp.distributions.Distribution] = None,
    ) -> None:
        super().__init__(
            feature_dims=feature_dims,
            feature_bounds=feature_bounds,
            indicator_dims=indicator_dims,
            activity_conditions=activity_conditions,
            base_kernel=base_kernel,
        )

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

    def _embed_conditional(
        self, cond_v: tf.Tensor, cond_mask: tf.Tensor
    ) -> list[tf.Tensor]:
        theta = _PI * self.angle * cond_v
        sin_part = self.radius * tf.sin(theta) * cond_mask
        cos_part = self.radius * tf.cos(theta) * cond_mask
        return [sin_part, cos_part]


# ---------------------------------------------------------------------------
# WedgeKernel
# ---------------------------------------------------------------------------


class WedgeKernel(_HierarchicalEmbeddingKernel):
    """Triangular wedge kernel for hierarchical search spaces (Horn et al., 2019).

    Each conditional feature dimension is mapped to R^2 via::

        active:   [theta1 * v + theta2 * v * cos(rho),  theta2 * v * sin(rho)]
        inactive: [0, 0]

    where ``v = (x - lower) / (upper - lower)`` is the normalised value.
    Unlike the arc kernel, the incomparable distance (one active, one
    inactive) depends on the active-side value, which is strictly more
    informative near disjunction boundaries.

    :param feature_dims: Indices into the flat input ``X`` giving the
        real-valued (non-indicator) feature columns, in the order matching
        ``feature_bounds`` and ``activity_conditions``.
    :param feature_bounds: ``[D_f, 2]`` tensor of ``(lower, upper)`` pairs
        for each feature dimension.
    :param indicator_dims: Indices into ``X`` of the 0/1 indicator columns
        used by ``activity_conditions``.
    :param activity_conditions: One entry per feature dimension giving a
        sequence of ``(indicator_local_index, required_bool)`` pairs that
        must all hold for the feature to be active.
    :param base_kernel: GPflow kernel applied in the embedded space.
    :param theta1_prior: Optional prior on the ``theta1`` parameter.
    :param theta2_prior: Optional prior on the ``theta2`` parameter.
    :param rho_prior: Optional prior on the ``rho`` parameter.
    """

    def __init__(
        self,
        feature_dims: Sequence[int],
        feature_bounds: TensorType,
        indicator_dims: Sequence[int] = (),
        activity_conditions: Sequence[Sequence[Tuple[int, bool]]] = (),
        base_kernel: Optional[gpflow.kernels.Kernel] = None,
        theta1_prior: Optional[tfp.distributions.Distribution] = None,
        theta2_prior: Optional[tfp.distributions.Distribution] = None,
        rho_prior: Optional[tfp.distributions.Distribution] = None,
    ) -> None:
        super().__init__(
            feature_dims=feature_dims,
            feature_bounds=feature_bounds,
            indicator_dims=indicator_dims,
            activity_conditions=activity_conditions,
            base_kernel=base_kernel,
        )

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

    def _embed_conditional(
        self, cond_v: tf.Tensor, cond_mask: tf.Tensor
    ) -> list[tf.Tensor]:
        comp1 = (self.theta1 * cond_v + self.theta2 * cond_v * tf.cos(self.rho)) * cond_mask
        comp2 = (self.theta2 * cond_v * tf.sin(self.rho)) * cond_mask
        return [comp1, comp2]
