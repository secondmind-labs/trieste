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
"""Tests for the Trieste-side factory that builds :class:`ArcKernel` /
:class:`WedgeKernel` from a :class:`HierarchicalSearchSpace`."""
from __future__ import annotations

import gpflow
import numpy as np
import numpy.testing as npt
import pytest
import tensorflow as tf

from trieste.models.gpflow.kernels.hierarchical import ArcKernel, WedgeKernel
from trieste.models.gpflow.kernels.hierarchical_builders import (
    arc_kernel_from_space,
    primitives_from_space,
    wedge_kernel_from_space,
)
from trieste.space import (
    BooleanSearchSpace,
    Box,
    HierarchicalSearchSpace,
    HierarchyNode,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _simple_space() -> HierarchicalSearchSpace:
    return HierarchicalSearchSpace(
        spaces=[Box([0.0], [1.0]), BooleanSearchSpace(), Box([0.0], [5.0])],
        tags=["x1", "y1", "x2"],
        hierarchy=[
            HierarchyNode("shared", subspace_tags=["x1"], indicator_conditions={}),
            HierarchyNode(
                "branch_A", subspace_tags=["x2"], indicator_conditions={"y1": True}
            ),
        ],
        indicator_tags=["y1"],
    )


def _two_indicator_space() -> HierarchicalSearchSpace:
    return HierarchicalSearchSpace(
        spaces=[
            Box([0.0], [1.0]),
            BooleanSearchSpace(),
            BooleanSearchSpace(),
            Box([0.0], [5.0]),
            Box([-1.0], [1.0]),
        ],
        tags=["x1", "y1", "y2", "x2", "x3"],
        hierarchy=[
            HierarchyNode("shared", subspace_tags=["x1"], indicator_conditions={}),
            HierarchyNode(
                "branch_A", subspace_tags=["x2"], indicator_conditions={"y1": True}
            ),
            HierarchyNode(
                "branch_B", subspace_tags=["x3"], indicator_conditions={"y2": True}
            ),
        ],
        indicator_tags=["y1", "y2"],
    )


def _multi_dim_space() -> HierarchicalSearchSpace:
    return HierarchicalSearchSpace(
        spaces=[
            Box([0.0, 0.0], [1.0, 1.0]),
            BooleanSearchSpace(),
            Box([0.0, 0.0], [5.0, 5.0]),
        ],
        tags=["x12", "y1", "x34"],
        hierarchy=[
            HierarchyNode("shared", subspace_tags=["x12"], indicator_conditions={}),
            HierarchyNode(
                "branch", subspace_tags=["x34"], indicator_conditions={"y1": True}
            ),
        ],
        indicator_tags=["y1"],
    )


# ---------------------------------------------------------------------------
# primitives_from_space
# ---------------------------------------------------------------------------


class TestPrimitivesFromSpace:
    def test_simple(self) -> None:
        space = _simple_space()
        feat_dims, bounds, ind_dims, conds = primitives_from_space(space)
        assert feat_dims == [0, 2]
        assert ind_dims == [1]
        npt.assert_allclose(bounds.numpy(), [[0.0, 1.0], [0.0, 5.0]])
        assert conds == [[], [(0, True)]]

    def test_two_indicators(self) -> None:
        space = _two_indicator_space()
        feat_dims, bounds, ind_dims, conds = primitives_from_space(space)
        assert feat_dims == [0, 3, 4]
        assert ind_dims == [1, 2]
        npt.assert_allclose(bounds.numpy(), [[0.0, 1.0], [0.0, 5.0], [-1.0, 1.0]])
        assert conds == [[], [(0, True)], [(1, True)]]

    def test_multi_dim_broadcasts_conditions(self) -> None:
        space = _multi_dim_space()
        feat_dims, bounds, ind_dims, conds = primitives_from_space(space)
        assert feat_dims == [0, 1, 3, 4]
        assert ind_dims == [2]
        npt.assert_allclose(
            bounds.numpy(),
            [[0.0, 1.0], [0.0, 1.0], [0.0, 5.0], [0.0, 5.0]],
        )
        assert conds == [[], [], [(0, True)], [(0, True)]]


# ---------------------------------------------------------------------------
# Factory equivalence to hand-built kernels
# ---------------------------------------------------------------------------


def _hand_built(kernel_cls, space: HierarchicalSearchSpace):
    feat_dims, bounds, ind_dims, conds = primitives_from_space(space)
    return kernel_cls(
        feature_dims=feat_dims,
        feature_bounds=bounds,
        indicator_dims=ind_dims,
        activity_conditions=conds,
    )


def _copy_trainables(src, dst) -> None:
    """Copy trainable parameter values from ``src`` to ``dst`` (ignoring the
    base kernel's frozen lengthscale)."""
    for a, b in zip(src.trainable_variables, dst.trainable_variables):
        b.assign(a)


class TestFactoryEquivalence:
    @pytest.mark.parametrize(
        "factory, kernel_cls",
        [(arc_kernel_from_space, ArcKernel), (wedge_kernel_from_space, WedgeKernel)],
    )
    def test_simple_numerical_equivalence(self, factory, kernel_cls) -> None:
        space = _simple_space()
        k_factory = factory(space)
        k_manual = _hand_built(kernel_cls, space)
        _copy_trainables(k_factory, k_manual)

        X = tf.constant(
            [[0.5, 1.0, 2.5], [0.3, 0.0, 1.0], [0.7, 1.0, 4.0]],
            dtype=tf.float64,
        )
        npt.assert_allclose(
            k_factory.K(X).numpy(), k_manual.K(X).numpy(), atol=1e-10
        )

    @pytest.mark.parametrize(
        "factory, kernel_cls",
        [(arc_kernel_from_space, ArcKernel), (wedge_kernel_from_space, WedgeKernel)],
    )
    def test_multi_dim_numerical_equivalence(self, factory, kernel_cls) -> None:
        space = _multi_dim_space()
        k_factory = factory(space)
        k_manual = _hand_built(kernel_cls, space)
        _copy_trainables(k_factory, k_manual)

        X = tf.constant(
            [
                [0.1, 0.2, 1.0, 2.5, 2.5],
                [0.3, 0.4, 0.0, 2.5, 2.5],
                [0.5, 0.6, 1.0, 1.0, 4.0],
            ],
            dtype=tf.float64,
        )
        npt.assert_allclose(
            k_factory.K(X).numpy(), k_manual.K(X).numpy(), atol=1e-10
        )

    def test_factory_accepts_custom_base_kernel(self) -> None:
        space = _simple_space()
        k = arc_kernel_from_space(
            space, base_kernel=gpflow.kernels.SquaredExponential()
        )
        X = tf.constant(
            [[0.2, 1.0, 1.0], [0.6, 0.0, 2.0]], dtype=tf.float64
        )
        K = k.K(X)
        assert K.shape == (2, 2)
        assert tf.reduce_all(tf.math.is_finite(K)).numpy()


# ---------------------------------------------------------------------------
# Full integration: GPR wrapped in a factory-built kernel
# ---------------------------------------------------------------------------


class TestGPRIntegration:
    @pytest.mark.parametrize(
        "factory", [arc_kernel_from_space, wedge_kernel_from_space]
    )
    def test_gpr_predict(self, factory) -> None:
        space = _simple_space()
        kernel = gpflow.kernels.Constant() * factory(space)

        np.random.seed(7)
        X_train = np.array(
            [
                [0.2, 1.0, 1.0],
                [0.4, 1.0, 3.0],
                [0.6, 0.0, 2.0],
                [0.8, 0.0, 4.0],
                [0.1, 1.0, 0.5],
            ]
        )
        Y_train = np.random.randn(5, 1)

        gpr = gpflow.models.GPR(
            data=(X_train, Y_train), kernel=kernel, noise_variance=0.1
        )
        X_test = np.array([[0.3, 1.0, 2.0], [0.5, 0.0, 1.5]])
        mean, var = gpr.predict_f(X_test)

        assert mean.shape == (2, 1)
        assert var.shape == (2, 1)
        assert tf.reduce_all(var > 0.0).numpy()
        assert tf.reduce_all(tf.math.is_finite(mean)).numpy()

    def test_hyperparameter_optimisation_moves_params(self) -> None:
        space = _simple_space()
        arc = arc_kernel_from_space(space)
        kernel = gpflow.kernels.Constant() * arc

        np.random.seed(0)
        X = np.random.uniform(size=(12, 3))
        X[:, 1] = (X[:, 1] > 0.5).astype(float)
        Y = np.sin(3.0 * X[:, 0:1]) + 0.3 * np.random.randn(12, 1)

        gpr = gpflow.models.GPR(data=(X, Y), kernel=kernel, noise_variance=0.1)

        angle_before = arc.angle.numpy().copy()
        radius_before = arc.radius.numpy().copy()

        opt = gpflow.optimizers.Scipy()
        opt.minimize(
            gpr.training_loss,
            gpr.trainable_variables,
            options={"maxiter": 25},
        )

        assert not np.allclose(angle_before, arc.angle.numpy()) or not np.allclose(
            radius_before, arc.radius.numpy()
        )
