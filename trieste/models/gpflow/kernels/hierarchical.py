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
list of :class:`ActivityCondition` objects that each bind an indicator-logic
conjunction to a specific feature column), so the kernels can be migrated
wholesale into GPflow in a later change.

References:
    - Swersky et al. (2014) -- Arc (cylindrical) kernel
    - Horn et al. (2019)    -- Wedge (triangular) kernel
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence, Tuple

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
# ActivityCondition: formal type binding a feature column to its AND-conjunction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActivityCondition:
    """AND-conjunction of indicator requirements for a single feature column.

    :param feature_dim: The column index into the flat input vector ``X``
        that this condition applies to. Must appear in the kernel's
        ``feature_dims`` argument.
    :param requirements: Mapping ``{indicator_local_index: required_value}``.
        The required value is a non-negative integer matching the indicator's
        permitted set: 0 or 1 for boolean indicators, or any value in
        ``0..K-1`` for a K-ary categorical indicator. All listed indicators
        must take their required value for the feature to be active. An
        empty mapping means the feature is unconditional.
    """

    feature_dim: int
    requirements: Mapping[int, int] = field(default_factory=dict)

    @classmethod
    def unconditional(cls, feature_dim: int) -> "ActivityCondition":
        """Return an unconditional activity condition for ``feature_dim``."""
        return cls(feature_dim=feature_dim, requirements={})

    @property
    def is_unconditional(self) -> bool:
        """True when no indicator requirements are attached."""
        return not self.requirements

    def items(self):
        """Iterate over ``(indicator_local_index, required_value)`` pairs."""
        return self.requirements.items()

    def __iter__(self):
        return iter(self.requirements)

    def __bool__(self) -> bool:
        return bool(self.requirements)


# ---------------------------------------------------------------------------
# Shared primitive helpers
# ---------------------------------------------------------------------------


def _compile_activity_conditions(
    activity_conditions: Sequence[ActivityCondition],
    feature_dims: Sequence[int],
    n_indicators: int,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Compile per-feature :class:`ActivityCondition` objects into dense tensors.

    Each :class:`ActivityCondition` binds its requirements to a specific
    ``feature_dim`` (column in ``X``). Features in ``feature_dims`` that are
    not referenced by any condition are treated as unconditional.

    :param activity_conditions: Sequence of :class:`ActivityCondition`
        objects. Each ``c.feature_dim`` must appear in ``feature_dims``, and
        no two objects may share the same ``feature_dim``.
    :param feature_dims: Column indices of real-valued features in ``X``, in
        the order that determines the local feature-position within the
        compiled tensors.
    :param n_indicators: Number of indicator dimensions ``D_i``.
    :return: Tuple ``(required, is_ignore)`` where ``required`` is an
        ``int32`` tensor of shape ``[D_f, D_i]`` with entries in
        ``{-1} ∪ {0, 1, ...}`` (``-1`` is the ignored-indicator sentinel;
        non-negative values are the required indicator values, supporting
        boolean ``0/1`` indicators and ``K``-ary categorical
        ``0..K-1`` indicators alike) and ``is_ignore`` is a ``bool`` tensor
        of the same shape, True where ``required == -1``.
    """
    n_features = len(feature_dims)
    feat_to_local = {int(dim): j for j, dim in enumerate(feature_dims)}
    if len(feat_to_local) != n_features:
        raise ValueError(
            f"feature_dims contains duplicate column indices: {list(feature_dims)}."
        )

    required = np.full((n_features, n_indicators), _IGNORE, dtype=np.int32)
    seen: set[int] = set()

    for c in activity_conditions:
        if not isinstance(c, ActivityCondition):
            raise TypeError(
                f"activity_conditions entries must be ActivityCondition instances; "
                f"got {type(c).__name__}."
            )
        if c.feature_dim not in feat_to_local:
            raise ValueError(
                f"ActivityCondition.feature_dim={c.feature_dim} is not in "
                f"feature_dims {list(feature_dims)}."
            )
        if c.feature_dim in seen:
            raise ValueError(
                f"Duplicate ActivityCondition for feature_dim={c.feature_dim}."
            )
        seen.add(c.feature_dim)

        j = feat_to_local[c.feature_dim]
        for k, required_value in c.items():
            if not (0 <= int(k) < n_indicators):
                raise ValueError(
                    f"ActivityCondition(feature_dim={c.feature_dim}) references "
                    f"indicator index {k}; must be in [0, {n_indicators})."
                )
            val = int(required_value)
            if val < 0:
                raise ValueError(
                    f"ActivityCondition(feature_dim={c.feature_dim}) requires "
                    f"value {required_value!r} for indicator {k}; required "
                    f"values must be non-negative integers (negative collides "
                    f"with the ignored-indicator sentinel)."
                )
            existing = required[j, int(k)]
            if existing != _IGNORE and existing != val:
                raise ValueError(
                    f"ActivityCondition(feature_dim={c.feature_dim}) contains "
                    f"contradictory requirements for indicator {k}: both "
                    f"{int(existing)} and {val}."
                )
            required[j, int(k)] = val

    required_t = tf.constant(required, dtype=tf.int32)
    is_ignore_t = tf.equal(required_t, _IGNORE)
    return required_t, is_ignore_t


def _classify_conditional(
    activity_conditions: Sequence[ActivityCondition],
    feature_dims: Sequence[int],
) -> Tuple[list[int], list[int]]:
    """Split feature columns into unconditional and conditional local indices.

    A feature column is *conditional* iff an :class:`ActivityCondition` with a
    matching ``feature_dim`` and non-empty ``requirements`` appears in
    ``activity_conditions``. Otherwise the feature is unconditional (either
    absent from the sequence or listed with empty requirements).

    :param activity_conditions: Sequence of :class:`ActivityCondition`
        objects with distinct ``feature_dim`` values drawn from
        ``feature_dims``.
    :param feature_dims: Column indices of real-valued features in ``X``, in
        the order defining local positions.
    :return: Tuple ``(unconditional_local_idx, conditional_local_idx)``
        listing positions within the feature-vector.
    """
    conditional_dims = {
        int(c.feature_dim) for c in activity_conditions if not c.is_unconditional
    }
    uncond: list[int] = []
    cond: list[int] = []
    for j, dim in enumerate(feature_dims):
        if int(dim) in conditional_dims:
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
        activity_conditions: Sequence[ActivityCondition] = (),
        base_kernel: Optional[gpflow.kernels.Kernel] = None,
    ) -> None:
        super().__init__()

        feature_dims = list(feature_dims)
        indicator_dims = list(indicator_dims)
        activity_conditions = list(activity_conditions)
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

        self._feature_dims = tf.constant(feature_dims, dtype=tf.int32)
        self._indicator_dims = tf.constant(indicator_dims, dtype=tf.int32)
        self._bounds = bounds_tensor
        self._n_feat = n_feat
        self._n_ind = n_ind

        self._required, self._required_is_ignore = _compile_activity_conditions(
            activity_conditions, feature_dims, n_ind
        )

        uncond_local, cond_local = _classify_conditional(
            activity_conditions, feature_dims
        )
        self._uncond_local_idx = uncond_local
        self._cond_local_idx = cond_local
        self._n_uncond = len(uncond_local)
        self._n_cond = len(cond_local)

        if base_kernel is None:
            base_kernel = gpflow.kernels.Matern52()
        if not isinstance(base_kernel, gpflow.kernels.Stationary):
            raise ValueError(
                f"base_kernel must be a gpflow.kernels.Stationary instance; "
                f"got {type(base_kernel).__name__}. The embedding geometry "
                f"relies on k(x, x') depending only on x - x': non-stationary "
                f"kernels break the both-inactive / both-active / incomparable "
                f"distance axioms."
            )
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
        ind_int = tf.cast(tf.round(ind_vals), tf.int32)  # [N, D_i]

        # Evaluate the per-feature AND-conjunction over all indicators by
        # broadcasting the batch of indicator values ``ind_int`` ([N, 1, D_i])
        # against the compiled requirements ``_required`` ([1, D_f, D_i]).
        # An indicator slot matches when either it is ignored for this feature
        # (``_required_is_ignore`` true) or the observed integer value equals
        # the required value (boolean indicators carry 0/1, categorical
        # indicators carry 0..K-1). A feature is active iff every indicator
        # matches.
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
        ``feature_bounds``. This order determines the local feature position
        used when building the embedded vector.
    :param feature_bounds: ``[D_f, 2]`` tensor of ``(lower, upper)`` pairs
        for each feature dimension.
    :param indicator_dims: Indices into ``X`` of the 0/1 indicator columns
        referenced by :class:`ActivityCondition.requirements` via their
        local position (``0 <= k < len(indicator_dims)``).
    :param activity_conditions: Sequence of :class:`ActivityCondition`
        objects. Each binds an AND-conjunction of indicator requirements to
        its target ``feature_dim``. Feature columns not referenced by any
        condition are treated as unconditional.
    :param base_kernel: GPflow stationary kernel applied in the embedded
        space. Must be a :class:`gpflow.kernels.Stationary` subclass (e.g.
        :class:`~gpflow.kernels.Matern52`,
        :class:`~gpflow.kernels.SquaredExponential`,
        :class:`~gpflow.kernels.RationalQuadratic`); its ``lengthscales`` are
        frozen at 1.0 so distance scaling stays with the embedding parameters.
        Non-stationary kernels (e.g. ``Linear``, ``Polynomial``, ``Constant``)
        are rejected because the embedding axioms rely on ``k(x, x')``
        depending only on ``x - x'``.
    :param angle_prior: Optional prior on the ``angle`` parameter.
    :param radius_prior: Optional prior on the ``radius`` parameter.
    """

    def __init__(
        self,
        feature_dims: Sequence[int],
        feature_bounds: TensorType,
        indicator_dims: Sequence[int] = (),
        activity_conditions: Sequence[ActivityCondition] = (),
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
        ``feature_bounds``.
    :param feature_bounds: ``[D_f, 2]`` tensor of ``(lower, upper)`` pairs
        for each feature dimension.
    :param indicator_dims: Indices into ``X`` of the 0/1 indicator columns
        referenced by :class:`ActivityCondition.requirements` via their
        local position.
    :param activity_conditions: Sequence of :class:`ActivityCondition`
        objects; see :class:`ArcKernel` for detailed semantics.
    :param base_kernel: GPflow stationary kernel applied in the embedded
        space. Must be a :class:`gpflow.kernels.Stationary` subclass (e.g.
        :class:`~gpflow.kernels.Matern52`,
        :class:`~gpflow.kernels.SquaredExponential`,
        :class:`~gpflow.kernels.RationalQuadratic`); its ``lengthscales`` are
        frozen at 1.0 so distance scaling stays with the embedding parameters.
        Non-stationary kernels (e.g. ``Linear``, ``Polynomial``, ``Constant``)
        are rejected because the embedding axioms rely on ``k(x, x')``
        depending only on ``x - x'``.
    :param theta1_prior: Optional prior on the ``theta1`` parameter.
    :param theta2_prior: Optional prior on the ``theta2`` parameter.
    :param rho_prior: Optional prior on the ``rho`` parameter.
    """

    def __init__(
        self,
        feature_dims: Sequence[int],
        feature_bounds: TensorType,
        indicator_dims: Sequence[int] = (),
        activity_conditions: Sequence[ActivityCondition] = (),
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
