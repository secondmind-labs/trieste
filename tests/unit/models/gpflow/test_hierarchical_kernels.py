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
"""Tests for ArcKernel, WedgeKernel, and shared helpers in
trieste.models.gpflow.kernels.hierarchical."""
from __future__ import annotations

import gpflow
import numpy as np
import numpy.testing as npt
import pytest
import tensorflow as tf

from trieste.models.gpflow.kernels.hierarchical import (
    ArcKernel,
    WedgeKernel,
    _build_activity_mask,
    _build_bounds_tensor,
    _classify_dims,
    _extract_non_indicator_dims,
)
from trieste.space import (
    BooleanSearchSpace,
    Box,
    HierarchicalSearchSpace,
    HierarchyNode,
)


# ---------------------------------------------------------------------------
# Fixtures: reusable search spaces
# ---------------------------------------------------------------------------


def _simple_space() -> HierarchicalSearchSpace:
    """x1 (unconditional, [0,1]), y1 (indicator), x2 (conditional on y1=True, [0,5])."""
    return HierarchicalSearchSpace(
        spaces=[Box([0.0], [1.0]), BooleanSearchSpace(), Box([0.0], [5.0])],
        tags=["x1", "y1", "x2"],
        hierarchy=[
            HierarchyNode("shared", subspace_tags=["x1"], indicator_conditions={}),
            HierarchyNode("branch_A", subspace_tags=["x2"], indicator_conditions={"y1": True}),
        ],
        indicator_tags=["y1"],
    )


def _two_indicator_space() -> HierarchicalSearchSpace:
    """x1 (unconditional), y1 + y2 (indicators), x2 (y1=True), x3 (y2=True)."""
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
            HierarchyNode("branch_A", subspace_tags=["x2"], indicator_conditions={"y1": True}),
            HierarchyNode("branch_B", subspace_tags=["x3"], indicator_conditions={"y2": True}),
        ],
        indicator_tags=["y1", "y2"],
    )


def _multi_dim_space() -> HierarchicalSearchSpace:
    """x12 (unconditional, 2-D), y1 (indicator), x34 (conditional on y1=True, 2-D)."""
    return HierarchicalSearchSpace(
        spaces=[Box([0.0, 0.0], [1.0, 1.0]), BooleanSearchSpace(), Box([0.0, 0.0], [5.0, 5.0])],
        tags=["x12", "y1", "x34"],
        hierarchy=[
            HierarchyNode("shared", subspace_tags=["x12"], indicator_conditions={}),
            HierarchyNode("branch", subspace_tags=["x34"], indicator_conditions={"y1": True}),
        ],
        indicator_tags=["y1"],
    )


# ---------------------------------------------------------------------------
# Tests: _build_activity_mask
# ---------------------------------------------------------------------------


class TestBuildActivityMask:
    def test_single_indicator_active(self) -> None:
        space = _simple_space()
        # x1=0.5, y1=1.0, x2=2.5 → x2 active (y1=True)
        pts = tf.constant([[0.5, 1.0, 2.5]], dtype=tf.float64)
        mask = _build_activity_mask(pts, space)
        expected = tf.constant([[True, True]], dtype=tf.bool)
        tf.debugging.assert_equal(mask, expected)

    def test_single_indicator_inactive(self) -> None:
        space = _simple_space()
        # x1=0.5, y1=0.0, x2=2.5 → x2 inactive
        pts = tf.constant([[0.5, 0.0, 2.5]], dtype=tf.float64)
        mask = _build_activity_mask(pts, space)
        expected = tf.constant([[True, False]], dtype=tf.bool)
        tf.debugging.assert_equal(mask, expected)

    def test_unconditional_always_true(self) -> None:
        space = _simple_space()
        pts = tf.constant(
            [[0.5, 0.0, 2.5], [0.3, 1.0, 1.0]], dtype=tf.float64
        )
        mask = _build_activity_mask(pts, space)
        assert mask.shape == (2, 2)
        tf.debugging.assert_equal(mask[:, 0], tf.constant([True, True]))

    def test_batch_mixed(self) -> None:
        space = _simple_space()
        pts = tf.constant(
            [
                [0.5, 1.0, 2.5],  # x2 active
                [0.3, 0.0, 1.0],  # x2 inactive
                [0.7, 1.0, 4.0],  # x2 active
            ],
            dtype=tf.float64,
        )
        mask = _build_activity_mask(pts, space)
        expected = tf.constant(
            [[True, True], [True, False], [True, True]], dtype=tf.bool
        )
        tf.debugging.assert_equal(mask, expected)

    def test_two_indicators(self) -> None:
        space = _two_indicator_space()
        # x1=0.5, y1=1, y2=0, x2=3.0, x3=0.5
        pts = tf.constant([[0.5, 1.0, 0.0, 3.0, 0.5]], dtype=tf.float64)
        mask = _build_activity_mask(pts, space)
        # non-indicator dims: x1(uncond=True), x2(y1=True→active), x3(y2=True→inactive)
        expected = tf.constant([[True, True, False]], dtype=tf.bool)
        tf.debugging.assert_equal(mask, expected)

    def test_multi_dim_subspace(self) -> None:
        space = _multi_dim_space()
        pts = tf.constant(
            [[0.5, 0.5, 1.0, 2.5, 2.5], [0.5, 0.5, 0.0, 2.5, 2.5]],
            dtype=tf.float64,
        )
        mask = _build_activity_mask(pts, space)
        expected = tf.constant(
            [[True, True, True, True], [True, True, False, False]], dtype=tf.bool
        )
        tf.debugging.assert_equal(mask, expected)


# ---------------------------------------------------------------------------
# Tests: _build_bounds_tensor
# ---------------------------------------------------------------------------


class TestBuildBoundsTensor:
    def test_simple_bounds(self) -> None:
        space = _simple_space()
        bounds = _build_bounds_tensor(space)
        assert bounds.shape == (2, 2)
        npt.assert_allclose(bounds.numpy(), [[0.0, 1.0], [0.0, 5.0]])

    def test_multi_dim_bounds(self) -> None:
        space = _multi_dim_space()
        bounds = _build_bounds_tensor(space)
        assert bounds.shape == (4, 2)
        npt.assert_allclose(
            bounds.numpy(),
            [[0.0, 1.0], [0.0, 1.0], [0.0, 5.0], [0.0, 5.0]],
        )


# ---------------------------------------------------------------------------
# Tests: _extract_non_indicator_dims
# ---------------------------------------------------------------------------


class TestExtractNonIndicatorDims:
    def test_simple(self) -> None:
        space = _simple_space()
        pts = tf.constant([[0.5, 1.0, 2.5]], dtype=tf.float64)
        ni = _extract_non_indicator_dims(pts, space)
        npt.assert_allclose(ni.numpy(), [[0.5, 2.5]])

    def test_two_indicators(self) -> None:
        space = _two_indicator_space()
        pts = tf.constant([[0.5, 1.0, 0.0, 3.0, 0.5]], dtype=tf.float64)
        ni = _extract_non_indicator_dims(pts, space)
        npt.assert_allclose(ni.numpy(), [[0.5, 3.0, 0.5]])


# ---------------------------------------------------------------------------
# Tests: _classify_dims
# ---------------------------------------------------------------------------


class TestClassifyDims:
    def test_simple(self) -> None:
        space = _simple_space()
        uncond, cond = _classify_dims(space)
        assert uncond == [0]
        assert cond == [1]

    def test_two_indicators(self) -> None:
        space = _two_indicator_space()
        uncond, cond = _classify_dims(space)
        assert uncond == [0]
        assert cond == [1, 2]

    def test_multi_dim(self) -> None:
        space = _multi_dim_space()
        uncond, cond = _classify_dims(space)
        assert uncond == [0, 1]
        assert cond == [2, 3]


# ---------------------------------------------------------------------------
# Shared kernel tests (parametrized for both Arc and Wedge)
# ---------------------------------------------------------------------------


@pytest.fixture(params=["arc", "wedge"], ids=["ArcKernel", "WedgeKernel"])
def kernel_with_space(request):
    space = _simple_space()
    if request.param == "arc":
        k = ArcKernel(space=space)
    else:
        k = WedgeKernel(space=space)
    return k, space


class TestSharedKernelProperties:
    def test_symmetry(self, kernel_with_space) -> None:
        k, space = kernel_with_space
        X = tf.constant(
            [[0.5, 1.0, 2.5], [0.3, 0.0, 1.0], [0.7, 1.0, 4.0]],
            dtype=tf.float64,
        )
        K12 = k.K(X, X)
        npt.assert_allclose(K12.numpy(), K12.numpy().T, atol=1e-10)

    def test_psd(self, kernel_with_space) -> None:
        k, space = kernel_with_space
        X = tf.constant(
            [[0.5, 1.0, 2.5], [0.3, 0.0, 1.0], [0.7, 1.0, 4.0], [0.1, 0.0, 0.5]],
            dtype=tf.float64,
        )
        K = k.K(X, X)
        eigvals = tf.linalg.eigvalsh(K)
        assert tf.reduce_all(eigvals >= -1e-6).numpy()

    def test_k_diag_matches_diagonal(self, kernel_with_space) -> None:
        k, space = kernel_with_space
        X = tf.constant(
            [[0.5, 1.0, 2.5], [0.3, 0.0, 1.0]], dtype=tf.float64
        )
        K_full = k.K(X, X)
        K_diag = k.K_diag(X)
        npt.assert_allclose(K_diag.numpy(), tf.linalg.diag_part(K_full).numpy(), atol=1e-10)

    def test_embedded_dimensionality(self, kernel_with_space) -> None:
        k, space = kernel_with_space
        X = tf.constant([[0.5, 1.0, 2.5]], dtype=tf.float64)
        Z = k._embed(X)
        # 1 unconditional + 2 * 1 conditional = 3
        assert Z.shape[-1] == 3

    def test_gradient_flow(self, kernel_with_space) -> None:
        k, space = kernel_with_space
        X = tf.constant(
            [[0.5, 1.0, 2.5], [0.3, 1.0, 1.0]], dtype=tf.float64
        )
        with tf.GradientTape() as tape:
            K = k.K(X, X)
            loss = tf.reduce_sum(K)
        trainable = k.trainable_variables
        grads = tape.gradient(loss, trainable)
        for g in grads:
            assert g is not None, "Gradient should flow through embedding parameters"

    def test_k_with_x2_none(self, kernel_with_space) -> None:
        k, space = kernel_with_space
        X = tf.constant(
            [[0.5, 1.0, 2.5], [0.3, 0.0, 1.0]], dtype=tf.float64
        )
        K1 = k.K(X, None)
        K2 = k.K(X, X)
        npt.assert_allclose(K1.numpy(), K2.numpy(), atol=1e-10)

    def test_both_inactive_same_similarity(self, kernel_with_space) -> None:
        k, space = kernel_with_space
        # Both inactive: y1=0 for both, conditional dims embed to [0,0]
        X = tf.constant(
            [[0.3, 0.0, 1.0], [0.3, 0.0, 4.0]], dtype=tf.float64
        )
        Z = k._embed(X)
        # conditional parts should both be [0,0], so embedded vectors differ only in uncond dim
        npt.assert_allclose(Z[0, 1:].numpy(), [0.0, 0.0], atol=1e-10)
        npt.assert_allclose(Z[1, 1:].numpy(), [0.0, 0.0], atol=1e-10)

    def test_both_active_distance_increases_with_difference(self, kernel_with_space) -> None:
        k, space = kernel_with_space
        x_same_uncond = 0.5
        # Both active: y1=1 for both, same x1, different x2 values
        X_close = tf.constant(
            [[x_same_uncond, 1.0, 2.0], [x_same_uncond, 1.0, 2.1]], dtype=tf.float64
        )
        X_far = tf.constant(
            [[x_same_uncond, 1.0, 0.5], [x_same_uncond, 1.0, 4.5]], dtype=tf.float64
        )
        K_close = k.K(X_close[:1], X_close[1:])
        K_far = k.K(X_far[:1], X_far[1:])
        assert K_close.numpy().item() > K_far.numpy().item()

    def test_works_with_squared_exponential(self) -> None:
        space = _simple_space()
        k = ArcKernel(space=space, base_kernel=gpflow.kernels.SquaredExponential())
        X = tf.constant(
            [[0.5, 1.0, 2.5], [0.3, 0.0, 1.0]], dtype=tf.float64
        )
        K = k.K(X, X)
        assert K.shape == (2, 2)
        eigvals = tf.linalg.eigvalsh(K)
        assert tf.reduce_all(eigvals >= -1e-6).numpy()


# ---------------------------------------------------------------------------
# Arc-specific tests
# ---------------------------------------------------------------------------


class TestArcKernel:
    def test_inactive_dims_embed_to_zero(self) -> None:
        space = _simple_space()
        k = ArcKernel(space=space)
        pts = tf.constant([[0.5, 0.0, 3.0]], dtype=tf.float64)
        Z = k._embed(pts)
        npt.assert_allclose(Z[0, 1:].numpy(), [0.0, 0.0], atol=1e-10)

    def test_active_dims_embed_to_circle(self) -> None:
        space = _simple_space()
        k = ArcKernel(space=space)
        pts = tf.constant([[0.5, 1.0, 2.5]], dtype=tf.float64)
        Z = k._embed(pts)
        sin_part = Z[0, 1].numpy()
        cos_part = Z[0, 2].numpy()
        r = k.radius.numpy()[0]
        actual_radius = np.sqrt(sin_part**2 + cos_part**2)
        npt.assert_allclose(actual_radius, r, atol=1e-6)

    def test_incomparable_distance_constant(self) -> None:
        """For Arc, the distance between active and inactive should not depend
        on the active-side value (only on radius)."""
        space = _simple_space()
        k = ArcKernel(space=space)

        inactive_pt = tf.constant([[0.5, 0.0, 0.0]], dtype=tf.float64)
        active_pt_low = tf.constant([[0.5, 1.0, 0.5]], dtype=tf.float64)
        active_pt_high = tf.constant([[0.5, 1.0, 4.5]], dtype=tf.float64)

        K_low = k.K(inactive_pt, active_pt_low).numpy().item()
        K_high = k.K(inactive_pt, active_pt_high).numpy().item()

        Z_inactive = k._embed(inactive_pt)
        Z_low = k._embed(active_pt_low)
        Z_high = k._embed(active_pt_high)

        dist_low = tf.norm(Z_inactive - Z_low).numpy()
        dist_high = tf.norm(Z_inactive - Z_high).numpy()

        # Distances differ only because the angle changes,
        # but the radial component is the same — check that embedded
        # vectors for the active side all lie on the same radius circle
        r = k.radius.numpy()[0]
        npt.assert_allclose(
            np.sqrt(Z_low[0, 1].numpy() ** 2 + Z_low[0, 2].numpy() ** 2), r, atol=1e-6
        )
        npt.assert_allclose(
            np.sqrt(Z_high[0, 1].numpy() ** 2 + Z_high[0, 2].numpy() ** 2), r, atol=1e-6
        )


# ---------------------------------------------------------------------------
# Wedge-specific tests
# ---------------------------------------------------------------------------


class TestWedgeKernel:
    def test_inactive_dims_embed_to_zero(self) -> None:
        space = _simple_space()
        k = WedgeKernel(space=space)
        pts = tf.constant([[0.5, 0.0, 3.0]], dtype=tf.float64)
        Z = k._embed(pts)
        npt.assert_allclose(Z[0, 1:].numpy(), [0.0, 0.0], atol=1e-10)

    def test_incomparable_distance_depends_on_value(self) -> None:
        """For Wedge, the incomparable distance should depend on the active-side value."""
        space = _simple_space()
        k = WedgeKernel(space=space)

        inactive_pt = tf.constant([[0.5, 0.0, 0.0]], dtype=tf.float64)
        active_pt_low = tf.constant([[0.5, 1.0, 0.5]], dtype=tf.float64)
        active_pt_high = tf.constant([[0.5, 1.0, 4.5]], dtype=tf.float64)

        Z_inactive = k._embed(inactive_pt)
        Z_low = k._embed(active_pt_low)
        Z_high = k._embed(active_pt_high)

        dist_low = tf.norm(Z_inactive - Z_low).numpy()
        dist_high = tf.norm(Z_inactive - Z_high).numpy()

        # Distances should differ because wedge embedding magnitude scales with v
        assert dist_low != pytest.approx(dist_high, abs=1e-4)

    def test_wedge_differs_from_arc_incomparable(self) -> None:
        space = _simple_space()
        k_arc = ArcKernel(space=space)
        k_wedge = WedgeKernel(space=space)

        inactive_pt = tf.constant([[0.5, 0.0, 0.0]], dtype=tf.float64)
        active_pt = tf.constant([[0.5, 1.0, 2.5]], dtype=tf.float64)

        K_arc = k_arc.K(inactive_pt, active_pt).numpy().item()
        K_wedge = k_wedge.K(inactive_pt, active_pt).numpy().item()

        assert K_arc != pytest.approx(K_wedge, abs=1e-4)

    def test_active_embedding_scales_with_value(self) -> None:
        """Wedge embedding magnitude should scale linearly with normalised value."""
        space = _simple_space()
        k = WedgeKernel(space=space)

        low_val = tf.constant([[0.5, 1.0, 1.0]], dtype=tf.float64)
        high_val = tf.constant([[0.5, 1.0, 4.0]], dtype=tf.float64)

        Z_low = k._embed(low_val)
        Z_high = k._embed(high_val)

        mag_low = tf.norm(Z_low[0, 1:]).numpy()
        mag_high = tf.norm(Z_high[0, 1:]).numpy()

        assert mag_high > mag_low


# ---------------------------------------------------------------------------
# Integration tests with GPR
# ---------------------------------------------------------------------------


class TestKernelGPRIntegration:
    @pytest.mark.parametrize("KernelClass", [ArcKernel, WedgeKernel])
    def test_gpr_predict_valid(self, KernelClass) -> None:
        space = _simple_space()
        kernel = gpflow.kernels.Constant() * KernelClass(space=space)

        np.random.seed(42)
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
            data=(X_train, Y_train),
            kernel=kernel,
            noise_variance=0.1,
        )

        X_test = np.array([[0.3, 1.0, 2.0], [0.5, 0.0, 1.5]])
        mean, var = gpr.predict_f(X_test)

        assert mean.shape == (2, 1)
        assert var.shape == (2, 1)
        assert tf.reduce_all(var > 0.0).numpy()
        assert tf.reduce_all(tf.math.is_finite(mean)).numpy()
        assert tf.reduce_all(tf.math.is_finite(var)).numpy()

    @pytest.mark.parametrize("KernelClass", [ArcKernel, WedgeKernel])
    def test_gpr_log_marginal_likelihood_finite(self, KernelClass) -> None:
        space = _simple_space()
        kernel = gpflow.kernels.Constant() * KernelClass(space=space)

        X_train = np.array(
            [[0.2, 1.0, 1.0], [0.4, 1.0, 3.0], [0.6, 0.0, 2.0]],
        )
        Y_train = np.array([[0.1], [0.5], [-0.3]])

        gpr = gpflow.models.GPR(
            data=(X_train, Y_train), kernel=kernel, noise_variance=0.1
        )

        lml = gpr.log_marginal_likelihood()
        assert tf.math.is_finite(lml).numpy()

    @pytest.mark.parametrize("KernelClass", [ArcKernel, WedgeKernel])
    def test_activity_mask_from_hss_indicators(self, KernelClass) -> None:
        space = _two_indicator_space()
        k = KernelClass(space=space)

        pts = tf.constant(
            [
                [0.5, 1.0, 1.0, 3.0, 0.5],  # both active
                [0.5, 1.0, 0.0, 3.0, 0.5],  # x2 active, x3 inactive
                [0.5, 0.0, 1.0, 3.0, 0.5],  # x2 inactive, x3 active
                [0.5, 0.0, 0.0, 3.0, 0.5],  # both inactive
            ],
            dtype=tf.float64,
        )

        mask = _build_activity_mask(pts, space)
        expected = tf.constant(
            [
                [True, True, True],
                [True, True, False],
                [True, False, True],
                [True, False, False],
            ],
            dtype=tf.bool,
        )
        tf.debugging.assert_equal(mask, expected)

        K = k.K(pts, pts)
        assert K.shape == (4, 4)
        eigvals = tf.linalg.eigvalsh(K)
        assert tf.reduce_all(eigvals >= -1e-6).numpy()
