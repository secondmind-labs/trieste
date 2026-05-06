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
"""Pure-gpflow tests for :class:`ArcKernel`, :class:`WedgeKernel`, and the
compact-mask helpers they use. No ``trieste.space`` symbols are imported here:
everything is driven by integer feature/indicator indices and an explicit
list of :class:`ActivityCondition` objects, mirroring the eventual
stand-alone GPflow API.
"""
from __future__ import annotations

from typing import List, Sequence

import gpflow
import numpy as np
import numpy.testing as npt
import pytest
import tensorflow as tf

from trieste.models.gpflow.kernels.hierarchical import (
    _IGNORE,
    ActivityCondition,
    ArcKernel,
    WedgeKernel,
    _classify_conditional,
    _compile_activity_conditions,
)


# ---------------------------------------------------------------------------
# Primitive fixtures mirroring the original HSS fixtures
# ---------------------------------------------------------------------------


def _simple_primitives() -> dict:
    """x1 at col 0 ([0,1], uncond); y1 at col 1; x2 at col 2 ([0,5], y1=True)."""
    return dict(
        feature_dims=[0, 2],
        feature_bounds=tf.constant([[0.0, 1.0], [0.0, 5.0]], dtype=tf.float64),
        indicator_dims=[1],
        activity_conditions=[
            ActivityCondition(feature_dim=0),
            ActivityCondition(feature_dim=2, requirements={0: True}),
        ],
    )


def _two_indicator_primitives() -> dict:
    """x1(col 0,uncond), y1(col 1), y2(col 2), x2(col 3,y1=True), x3(col 4,y2=True)."""
    return dict(
        feature_dims=[0, 3, 4],
        feature_bounds=tf.constant(
            [[0.0, 1.0], [0.0, 5.0], [-1.0, 1.0]], dtype=tf.float64
        ),
        indicator_dims=[1, 2],
        activity_conditions=[
            ActivityCondition(feature_dim=0),
            ActivityCondition(feature_dim=3, requirements={0: True}),
            ActivityCondition(feature_dim=4, requirements={1: True}),
        ],
    )


def _multi_dim_primitives() -> dict:
    """x12 (cols 0-1, uncond), y1 (col 2), x34 (cols 3-4, y1=True)."""
    return dict(
        feature_dims=[0, 1, 3, 4],
        feature_bounds=tf.constant(
            [[0.0, 1.0], [0.0, 1.0], [0.0, 5.0], [0.0, 5.0]], dtype=tf.float64
        ),
        indicator_dims=[2],
        activity_conditions=[
            ActivityCondition(feature_dim=0),
            ActivityCondition(feature_dim=1),
            ActivityCondition(feature_dim=3, requirements={0: True}),
            ActivityCondition(feature_dim=4, requirements={0: True}),
        ],
    )


def _and_primitives() -> dict:
    """Single feature conditional on AND of two indicators at cols 1 and 2."""
    return dict(
        feature_dims=[0, 3],
        feature_bounds=tf.constant([[0.0, 1.0], [0.0, 5.0]], dtype=tf.float64),
        indicator_dims=[1, 2],
        activity_conditions=[
            ActivityCondition(feature_dim=0),
            ActivityCondition(feature_dim=3, requirements={0: True, 1: False}),
        ],
    )


# ---------------------------------------------------------------------------
# ActivityCondition dataclass behaviour
# ---------------------------------------------------------------------------


class TestActivityCondition:
    def test_unconditional_constructor(self) -> None:
        c = ActivityCondition.unconditional(feature_dim=3)
        assert c.feature_dim == 3
        assert dict(c.requirements) == {}
        assert c.is_unconditional
        assert not bool(c)

    def test_with_requirements(self) -> None:
        c = ActivityCondition(feature_dim=1, requirements={0: True, 1: False})
        assert c.feature_dim == 1
        assert not c.is_unconditional
        assert bool(c)
        assert sorted(c.items()) == [(0, True), (1, False)]
        assert sorted(iter(c)) == [0, 1]

    def test_frozen(self) -> None:
        c = ActivityCondition(feature_dim=0, requirements={0: True})
        with pytest.raises(Exception):
            c.feature_dim = 1  # type: ignore[misc]

    def test_accepts_int_requirements(self) -> None:
        # Categorical indicators carry integer values >= 2; they must
        # round-trip through the dataclass intact (no bool coercion).
        c = ActivityCondition(feature_dim=0, requirements={0: 2})
        assert c.requirements[0] == 2
        assert not c.is_unconditional
        assert bool(c)


# ---------------------------------------------------------------------------
# Tests: _compile_activity_conditions
# ---------------------------------------------------------------------------


class TestCompileActivityConditions:
    def test_empty_conditions_all_ignore(self) -> None:
        required, is_ignore = _compile_activity_conditions([], [0, 5], 3)
        assert required.shape == (2, 3)
        assert tf.reduce_all(is_ignore).numpy()
        npt.assert_array_equal(required.numpy(), -np.ones((2, 3), dtype=np.int32))

    def test_single_and_condition(self) -> None:
        required, is_ignore = _compile_activity_conditions(
            [ActivityCondition(feature_dim=0, requirements={0: True})], [0], 2
        )
        assert required.numpy().tolist() == [[1, -1]]
        assert is_ignore.numpy().tolist() == [[False, True]]

    def test_multi_indicator_and(self) -> None:
        required, _ = _compile_activity_conditions(
            [ActivityCondition(feature_dim=7, requirements={0: True, 1: False})],
            [7],
            2,
        )
        assert required.numpy().tolist() == [[1, 0]]

    def test_omitted_feature_defaults_unconditional(self) -> None:
        required, is_ignore = _compile_activity_conditions(
            [ActivityCondition(feature_dim=2, requirements={0: True})],
            feature_dims=[0, 2],
            n_indicators=1,
        )
        assert required.numpy().tolist() == [[-1], [1]]
        assert is_ignore.numpy().tolist() == [[True], [False]]

    def test_unknown_feature_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="not in feature_dims"):
            _compile_activity_conditions(
                [ActivityCondition(feature_dim=99, requirements={0: True})],
                feature_dims=[0, 2],
                n_indicators=1,
            )

    def test_duplicate_feature_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="Duplicate ActivityCondition"):
            _compile_activity_conditions(
                [
                    ActivityCondition(feature_dim=0, requirements={0: True}),
                    ActivityCondition(feature_dim=0, requirements={0: False}),
                ],
                feature_dims=[0, 2],
                n_indicators=1,
            )

    def test_duplicate_feature_dims_input_raises(self) -> None:
        with pytest.raises(ValueError, match="duplicate column indices"):
            _compile_activity_conditions([], feature_dims=[0, 0], n_indicators=1)

    def test_non_condition_entry_raises(self) -> None:
        with pytest.raises(TypeError, match="ActivityCondition"):
            _compile_activity_conditions(
                [(0, True)],  # type: ignore[list-item]
                feature_dims=[0],
                n_indicators=1,
            )

    def test_out_of_range_indicator_raises(self) -> None:
        with pytest.raises(ValueError, match="references indicator index"):
            _compile_activity_conditions(
                [ActivityCondition(feature_dim=0, requirements={5: True})],
                feature_dims=[0],
                n_indicators=2,
            )

    def test_contradictory_requirements_raise(self) -> None:
        # Built directly because dict literals will dedupe duplicate keys.
        contradictory = ActivityCondition.__new__(ActivityCondition)
        # Use object.__setattr__ on the frozen dataclass for this test only.
        object.__setattr__(contradictory, "feature_dim", 0)

        class _Requirements(dict):
            def items(self):  # type: ignore[override]
                return [(0, True), (0, False)]

        object.__setattr__(contradictory, "requirements", _Requirements())
        with pytest.raises(ValueError, match="contradictory requirements"):
            _compile_activity_conditions(
                [contradictory], feature_dims=[0], n_indicators=1
            )

    def test_compiles_categorical_int_value(self) -> None:
        required, is_ignore = _compile_activity_conditions(
            [ActivityCondition(feature_dim=0, requirements={0: 2})],
            feature_dims=[0],
            n_indicators=1,
        )
        assert required.numpy().tolist() == [[2]]
        assert is_ignore.numpy().tolist() == [[False]]

    def test_rejects_negative_required_value(self) -> None:
        # A negative value would collide with the _IGNORE = -1 sentinel.
        with pytest.raises(ValueError, match="non-negative"):
            _compile_activity_conditions(
                [ActivityCondition(feature_dim=0, requirements={0: -1})],
                feature_dims=[0],
                n_indicators=1,
            )

    def test_contradiction_message_uses_int(self) -> None:
        # The error message should report the actual integer values, not
        # bool-coerced ones (which previously made "1 vs 2" read as "True
        # and True").
        contradictory = ActivityCondition.__new__(ActivityCondition)
        object.__setattr__(contradictory, "feature_dim", 0)

        class _Requirements(dict):
            def items(self):  # type: ignore[override]
                return [(0, 1), (0, 2)]

        object.__setattr__(contradictory, "requirements", _Requirements())
        with pytest.raises(ValueError, match="both 1 and 2"):
            _compile_activity_conditions(
                [contradictory], feature_dims=[0], n_indicators=1
            )


class TestClassifyConditional:
    def test_all_unconditional(self) -> None:
        uncond, cond = _classify_conditional([], [0, 2])
        assert uncond == [0, 1]
        assert cond == []

    def test_all_conditional(self) -> None:
        uncond, cond = _classify_conditional(
            [
                ActivityCondition(feature_dim=0, requirements={0: True}),
                ActivityCondition(feature_dim=2, requirements={0: False}),
            ],
            [0, 2],
        )
        assert uncond == []
        assert cond == [0, 1]

    def test_mixed(self) -> None:
        uncond, cond = _classify_conditional(
            [
                ActivityCondition(feature_dim=3, requirements={0: True}),
                ActivityCondition(feature_dim=7, requirements={1: True}),
            ],
            [0, 3, 5, 7],
        )
        assert uncond == [0, 2]
        assert cond == [1, 3]

    def test_unconditional_condition_is_unconditional(self) -> None:
        uncond, cond = _classify_conditional(
            [ActivityCondition(feature_dim=0)], [0, 1]
        )
        assert uncond == [0, 1]
        assert cond == []


# ---------------------------------------------------------------------------
# Round-trip: rebuild ActivityCondition list from compiled _required tensor
# ---------------------------------------------------------------------------


def _decode_required(
    required: tf.Tensor, feature_dims: Sequence[int]
) -> List[ActivityCondition]:
    req = required.numpy()
    out: List[ActivityCondition] = []
    for j, dim in enumerate(feature_dims):
        row = req[j]
        reqs = {int(k): bool(v) for k, v in enumerate(row) if int(v) != _IGNORE}
        out.append(ActivityCondition(feature_dim=int(dim), requirements=reqs))
    return out


class TestRoundTripFromCompiledTensor:
    def test_round_trip_preserves_conditions(self) -> None:
        feature_dims = [0, 3, 4]
        n_ind = 2
        conditions = [
            ActivityCondition(feature_dim=0),
            ActivityCondition(feature_dim=3, requirements={0: True}),
            ActivityCondition(feature_dim=4, requirements={1: False, 0: True}),
        ]
        required, _ = _compile_activity_conditions(conditions, feature_dims, n_ind)
        decoded = _decode_required(required, feature_dims)

        expected = {c.feature_dim: dict(c.requirements) for c in conditions}
        got = {c.feature_dim: dict(c.requirements) for c in decoded}
        assert got == expected

    def test_round_trip_from_kernel_attrs(self) -> None:
        k = ArcKernel(**_simple_primitives())
        decoded = _decode_required(k._required, [0, 2])
        assert decoded[0].feature_dim == 0
        assert dict(decoded[0].requirements) == {}
        assert decoded[1].feature_dim == 2
        assert dict(decoded[1].requirements) == {0: True}


# ---------------------------------------------------------------------------
# Tests: kernel-owned mask builder
# ---------------------------------------------------------------------------


def _mask(kernel_cls, primitives: dict, pts: tf.Tensor) -> tf.Tensor:
    k = kernel_cls(**primitives)
    return k._build_activity_mask(pts)


class TestBuildActivityMask:
    @pytest.mark.parametrize("kernel_cls", [ArcKernel, WedgeKernel])
    def test_single_indicator_active(self, kernel_cls) -> None:
        pts = tf.constant([[0.5, 1.0, 2.5]], dtype=tf.float64)
        mask = _mask(kernel_cls, _simple_primitives(), pts)
        tf.debugging.assert_equal(mask, tf.constant([[True, True]], dtype=tf.bool))

    @pytest.mark.parametrize("kernel_cls", [ArcKernel, WedgeKernel])
    def test_single_indicator_inactive(self, kernel_cls) -> None:
        pts = tf.constant([[0.5, 0.0, 2.5]], dtype=tf.float64)
        mask = _mask(kernel_cls, _simple_primitives(), pts)
        tf.debugging.assert_equal(mask, tf.constant([[True, False]], dtype=tf.bool))

    @pytest.mark.parametrize("kernel_cls", [ArcKernel, WedgeKernel])
    def test_unconditional_always_true(self, kernel_cls) -> None:
        pts = tf.constant(
            [[0.5, 0.0, 2.5], [0.3, 1.0, 1.0]], dtype=tf.float64
        )
        mask = _mask(kernel_cls, _simple_primitives(), pts)
        assert mask.shape == (2, 2)
        tf.debugging.assert_equal(mask[:, 0], tf.constant([True, True]))

    @pytest.mark.parametrize("kernel_cls", [ArcKernel, WedgeKernel])
    def test_batch_mixed(self, kernel_cls) -> None:
        pts = tf.constant(
            [[0.5, 1.0, 2.5], [0.3, 0.0, 1.0], [0.7, 1.0, 4.0]],
            dtype=tf.float64,
        )
        mask = _mask(kernel_cls, _simple_primitives(), pts)
        expected = tf.constant(
            [[True, True], [True, False], [True, True]], dtype=tf.bool
        )
        tf.debugging.assert_equal(mask, expected)

    @pytest.mark.parametrize("kernel_cls", [ArcKernel, WedgeKernel])
    def test_two_independent_indicators(self, kernel_cls) -> None:
        pts = tf.constant([[0.5, 1.0, 0.0, 3.0, 0.5]], dtype=tf.float64)
        mask = _mask(kernel_cls, _two_indicator_primitives(), pts)
        expected = tf.constant([[True, True, False]], dtype=tf.bool)
        tf.debugging.assert_equal(mask, expected)

    @pytest.mark.parametrize("kernel_cls", [ArcKernel, WedgeKernel])
    def test_and_over_two_indicators(self, kernel_cls) -> None:
        # condition: y1=True AND y2=False
        pts = tf.constant(
            [
                [0.5, 1.0, 0.0, 2.5],  # satisfies y1=T and y2=F -> active
                [0.5, 1.0, 1.0, 2.5],  # y2=T, violates AND -> inactive
                [0.5, 0.0, 0.0, 2.5],  # y1=F -> inactive
                [0.5, 0.0, 1.0, 2.5],  # both wrong -> inactive
            ],
            dtype=tf.float64,
        )
        mask = _mask(kernel_cls, _and_primitives(), pts)
        expected = tf.constant(
            [[True, True], [True, False], [True, False], [True, False]],
            dtype=tf.bool,
        )
        tf.debugging.assert_equal(mask, expected)

    @pytest.mark.parametrize("kernel_cls", [ArcKernel, WedgeKernel])
    def test_multi_dim_broadcast(self, kernel_cls) -> None:
        pts = tf.constant(
            [[0.5, 0.5, 1.0, 2.5, 2.5], [0.5, 0.5, 0.0, 2.5, 2.5]],
            dtype=tf.float64,
        )
        mask = _mask(kernel_cls, _multi_dim_primitives(), pts)
        expected = tf.constant(
            [[True, True, True, True], [True, True, False, False]], dtype=tf.bool
        )
        tf.debugging.assert_equal(mask, expected)

    @pytest.mark.parametrize("kernel_cls", [ArcKernel, WedgeKernel])
    def test_no_indicators_all_active(self, kernel_cls) -> None:
        prim = dict(
            feature_dims=[0, 1],
            feature_bounds=tf.constant([[0.0, 1.0], [0.0, 1.0]], dtype=tf.float64),
            indicator_dims=[],
            activity_conditions=[
                ActivityCondition(feature_dim=0),
                ActivityCondition(feature_dim=1),
            ],
        )
        pts = tf.constant([[0.1, 0.2], [0.3, 0.4]], dtype=tf.float64)
        mask = _mask(kernel_cls, prim, pts)
        tf.debugging.assert_equal(mask, tf.constant([[True, True], [True, True]]))

    @pytest.mark.parametrize("kernel_cls", [ArcKernel, WedgeKernel])
    def test_categorical_indicator_mask(self, kernel_cls) -> None:
        # x1 (col 0, uncond), y1 (col 1, 3-ary categorical), x2 (col 2, y1=2).
        prim = dict(
            feature_dims=[0, 2],
            feature_bounds=tf.constant([[0.0, 1.0], [0.0, 5.0]], dtype=tf.float64),
            indicator_dims=[1],
            activity_conditions=[
                ActivityCondition(feature_dim=0),
                ActivityCondition(feature_dim=2, requirements={0: 2}),
            ],
        )
        pts = tf.constant(
            [[0.1, 0.0, 0.5], [0.1, 1.0, 0.5], [0.1, 2.0, 0.5]], dtype=tf.float64
        )
        mask = _mask(kernel_cls, prim, pts)
        # x1 always active; x2 only when y1 == 2.
        expected = tf.constant(
            [[True, False], [True, False], [True, True]], dtype=tf.bool
        )
        tf.debugging.assert_equal(mask, expected)

    @pytest.mark.parametrize("kernel_cls", [ArcKernel, WedgeKernel])
    def test_categorical_mask_robust_to_float_drift(self, kernel_cls) -> None:
        # Indicator values stored as floats with small drift around the
        # integer category should still round to the right integer.
        prim = dict(
            feature_dims=[0, 2],
            feature_bounds=tf.constant([[0.0, 1.0], [0.0, 5.0]], dtype=tf.float64),
            indicator_dims=[1],
            activity_conditions=[
                ActivityCondition(feature_dim=0),
                ActivityCondition(feature_dim=2, requirements={0: 2}),
            ],
        )
        pts = tf.constant(
            [[0.1, 1.999, 0.5], [0.1, 2.001, 0.5]], dtype=tf.float64
        )
        mask = _mask(kernel_cls, prim, pts)
        # Both rows should round y1 to 2 and activate x2.
        expected = tf.constant([[True, True], [True, True]], dtype=tf.bool)
        tf.debugging.assert_equal(mask, expected)


# ---------------------------------------------------------------------------
# Omission + order invariance
# ---------------------------------------------------------------------------


class TestOmissionAndOrderSemantics:
    @pytest.mark.parametrize("kernel_cls", [ArcKernel, WedgeKernel])
    def test_omitting_unconditional_features_works(self, kernel_cls) -> None:
        full = kernel_cls(**_simple_primitives())
        sparse = kernel_cls(
            feature_dims=[0, 2],
            feature_bounds=tf.constant([[0.0, 1.0], [0.0, 5.0]], dtype=tf.float64),
            indicator_dims=[1],
            activity_conditions=[
                ActivityCondition(feature_dim=2, requirements={0: True}),
            ],
        )
        npt.assert_array_equal(full._required.numpy(), sparse._required.numpy())
        assert full._uncond_local_idx == sparse._uncond_local_idx
        assert full._cond_local_idx == sparse._cond_local_idx

    @pytest.mark.parametrize("kernel_cls", [ArcKernel, WedgeKernel])
    def test_condition_order_does_not_matter(self, kernel_cls) -> None:
        prim = _simple_primitives()
        reversed_conditions = list(reversed(prim["activity_conditions"]))
        reversed_prim = {**prim, "activity_conditions": reversed_conditions}

        k1 = kernel_cls(**prim)
        k2 = kernel_cls(**reversed_prim)

        X = tf.constant(
            [[0.5, 1.0, 2.5], [0.3, 0.0, 1.0], [0.7, 1.0, 4.0]], dtype=tf.float64
        )
        npt.assert_allclose(k1.K(X).numpy(), k2.K(X).numpy(), atol=1e-10)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestConstructorValidation:
    def test_bounds_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match=r"feature_bounds has \d+ rows"):
            ArcKernel(
                feature_dims=[0, 1],
                feature_bounds=tf.constant([[0.0, 1.0]], dtype=tf.float64),
                indicator_dims=[],
                activity_conditions=[
                    ActivityCondition(feature_dim=0),
                    ActivityCondition(feature_dim=1),
                ],
            )

    def test_bounds_wrong_inner_dim_raises(self) -> None:
        with pytest.raises(ValueError, match=r"feature_bounds must have shape"):
            ArcKernel(
                feature_dims=[0],
                feature_bounds=tf.constant([[0.0, 1.0, 2.0]], dtype=tf.float64),
                indicator_dims=[],
                activity_conditions=[ActivityCondition(feature_dim=0)],
            )

    def test_default_activity_conditions_are_unconditional(self) -> None:
        k = ArcKernel(
            feature_dims=[0, 1],
            feature_bounds=tf.constant([[0.0, 1.0], [0.0, 1.0]], dtype=tf.float64),
        )
        assert k._n_cond == 0
        assert k._n_uncond == 2

    def test_unknown_feature_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="not in feature_dims"):
            ArcKernel(
                feature_dims=[0, 2],
                feature_bounds=tf.constant([[0.0, 1.0], [0.0, 1.0]], dtype=tf.float64),
                indicator_dims=[1],
                activity_conditions=[
                    ActivityCondition(feature_dim=99, requirements={0: True}),
                ],
            )

    def test_duplicate_feature_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="Duplicate ActivityCondition"):
            ArcKernel(
                feature_dims=[0, 2],
                feature_bounds=tf.constant([[0.0, 1.0], [0.0, 1.0]], dtype=tf.float64),
                indicator_dims=[1],
                activity_conditions=[
                    ActivityCondition(feature_dim=0, requirements={0: True}),
                    ActivityCondition(feature_dim=0, requirements={0: False}),
                ],
            )

    @pytest.mark.parametrize("kernel_cls_", [ArcKernel, WedgeKernel])
    @pytest.mark.parametrize(
        "non_stationary_factory",
        [gpflow.kernels.Linear, gpflow.kernels.Polynomial, gpflow.kernels.Constant],
    )
    def test_non_stationary_base_kernel_is_rejected(
        self, kernel_cls_, non_stationary_factory
    ) -> None:
        with pytest.raises(ValueError, match="Stationary"):
            kernel_cls_(
                **_simple_primitives(), base_kernel=non_stationary_factory()
            )


# ---------------------------------------------------------------------------
# Shared kernel tests (parametrised for both Arc and Wedge)
# ---------------------------------------------------------------------------


@pytest.fixture(params=["arc", "wedge"], ids=["ArcKernel", "WedgeKernel"])
def kernel_cls(request):
    return ArcKernel if request.param == "arc" else WedgeKernel


@pytest.fixture
def simple_kernel(kernel_cls):
    return kernel_cls(**_simple_primitives())


class TestSharedKernelProperties:
    def test_symmetry(self, simple_kernel) -> None:
        X = tf.constant(
            [[0.5, 1.0, 2.5], [0.3, 0.0, 1.0], [0.7, 1.0, 4.0]],
            dtype=tf.float64,
        )
        K = simple_kernel.K(X, X)
        npt.assert_allclose(K.numpy(), K.numpy().T, atol=1e-10)

    def test_psd(self, simple_kernel) -> None:
        X = tf.constant(
            [[0.5, 1.0, 2.5], [0.3, 0.0, 1.0], [0.7, 1.0, 4.0], [0.1, 0.0, 0.5]],
            dtype=tf.float64,
        )
        K = simple_kernel.K(X, X)
        eigvals = tf.linalg.eigvalsh(K)
        assert tf.reduce_all(eigvals >= -1e-6).numpy()

    def test_k_diag_matches_diagonal(self, simple_kernel) -> None:
        X = tf.constant(
            [[0.5, 1.0, 2.5], [0.3, 0.0, 1.0]], dtype=tf.float64
        )
        K_full = simple_kernel.K(X, X)
        K_diag = simple_kernel.K_diag(X)
        npt.assert_allclose(K_diag.numpy(), tf.linalg.diag_part(K_full).numpy(), atol=1e-10)

    def test_embedded_dimensionality(self, simple_kernel) -> None:
        X = tf.constant([[0.5, 1.0, 2.5]], dtype=tf.float64)
        Z = simple_kernel._embed(X)
        # 1 unconditional + 2 * 1 conditional = 3
        assert Z.shape[-1] == 3

    def test_gradient_flow(self, simple_kernel) -> None:
        X = tf.constant(
            [[0.5, 1.0, 2.5], [0.3, 1.0, 1.0]], dtype=tf.float64
        )
        with tf.GradientTape() as tape:
            K = simple_kernel.K(X, X)
            loss = tf.reduce_sum(K)
        grads = tape.gradient(loss, simple_kernel.trainable_variables)
        assert grads, "kernel should have trainable variables"
        for g in grads:
            assert g is not None, "gradients must flow through embedding parameters"

    def test_k_with_x2_none(self, simple_kernel) -> None:
        X = tf.constant(
            [[0.5, 1.0, 2.5], [0.3, 0.0, 1.0]], dtype=tf.float64
        )
        K1 = simple_kernel.K(X, None)
        K2 = simple_kernel.K(X, X)
        npt.assert_allclose(K1.numpy(), K2.numpy(), atol=1e-10)

    def test_both_inactive_conditional_embed_to_zero(self, simple_kernel) -> None:
        X = tf.constant(
            [[0.3, 0.0, 1.0], [0.3, 0.0, 4.0]], dtype=tf.float64
        )
        Z = simple_kernel._embed(X)
        npt.assert_allclose(Z[0, 1:].numpy(), [0.0, 0.0], atol=1e-10)
        npt.assert_allclose(Z[1, 1:].numpy(), [0.0, 0.0], atol=1e-10)

    def test_both_active_distance_increases_with_difference(
        self, simple_kernel
    ) -> None:
        X_close = tf.constant(
            [[0.5, 1.0, 2.0], [0.5, 1.0, 2.1]], dtype=tf.float64
        )
        X_far = tf.constant(
            [[0.5, 1.0, 0.5], [0.5, 1.0, 4.5]], dtype=tf.float64
        )
        K_close = simple_kernel.K(X_close[:1], X_close[1:])
        K_far = simple_kernel.K(X_far[:1], X_far[1:])
        assert K_close.numpy().item() > K_far.numpy().item()


class TestBaseKernelInterchangeability:
    @pytest.mark.parametrize(
        "base",
        [gpflow.kernels.SquaredExponential, gpflow.kernels.Matern52],
    )
    @pytest.mark.parametrize("kernel_cls_", [ArcKernel, WedgeKernel])
    def test_alternative_base_kernels(self, base, kernel_cls_) -> None:
        k = kernel_cls_(**_simple_primitives(), base_kernel=base())
        X = tf.constant(
            [[0.5, 1.0, 2.5], [0.3, 0.0, 1.0]], dtype=tf.float64
        )
        K = k.K(X, X)
        assert K.shape == (2, 2)
        eigvals = tf.linalg.eigvalsh(K)
        assert tf.reduce_all(eigvals >= -1e-6).numpy()


class TestNoConditionalDegenerate:
    """When all features are unconditional, the kernel equals base_kernel on
    the normalised feature vector."""

    @pytest.mark.parametrize("kernel_cls_", [ArcKernel, WedgeKernel])
    def test_equals_base_kernel_on_normalised(self, kernel_cls_) -> None:
        bounds = tf.constant([[0.0, 1.0], [0.0, 4.0]], dtype=tf.float64)
        base = gpflow.kernels.Matern52()
        k = kernel_cls_(
            feature_dims=[0, 1],
            feature_bounds=bounds,
            indicator_dims=[],
            activity_conditions=[
                ActivityCondition(feature_dim=0),
                ActivityCondition(feature_dim=1),
            ],
            base_kernel=base,
        )
        X = tf.constant([[0.2, 1.0], [0.8, 3.0]], dtype=tf.float64)
        K = k.K(X).numpy()

        v = (X - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
        K_ref = base.K(v).numpy()
        npt.assert_allclose(K, K_ref, atol=1e-10)


# ---------------------------------------------------------------------------
# Arc-specific tests
# ---------------------------------------------------------------------------


class TestArcKernel:
    def test_inactive_dims_embed_to_zero(self) -> None:
        k = ArcKernel(**_simple_primitives())
        pts = tf.constant([[0.5, 0.0, 3.0]], dtype=tf.float64)
        Z = k._embed(pts)
        npt.assert_allclose(Z[0, 1:].numpy(), [0.0, 0.0], atol=1e-10)

    def test_active_dims_embed_to_circle(self) -> None:
        k = ArcKernel(**_simple_primitives())
        pts = tf.constant([[0.5, 1.0, 2.5]], dtype=tf.float64)
        Z = k._embed(pts)
        sin_part = Z[0, 1].numpy()
        cos_part = Z[0, 2].numpy()
        r = k.radius.numpy()[0]
        actual_radius = np.sqrt(sin_part**2 + cos_part**2)
        npt.assert_allclose(actual_radius, r, atol=1e-6)

    def test_incomparable_distance_constant_radius(self) -> None:
        """For Arc, every active-side embedding lies on a circle of radius
        ``radius`` regardless of the active-side value."""
        k = ArcKernel(**_simple_primitives())
        r = k.radius.numpy()[0]

        for x2 in (0.5, 2.5, 4.5):
            active = tf.constant([[0.5, 1.0, x2]], dtype=tf.float64)
            Z = k._embed(active)
            npt.assert_allclose(
                np.sqrt(Z[0, 1].numpy() ** 2 + Z[0, 2].numpy() ** 2), r, atol=1e-6
            )


# ---------------------------------------------------------------------------
# Wedge-specific tests
# ---------------------------------------------------------------------------


class TestWedgeKernel:
    def test_inactive_dims_embed_to_zero(self) -> None:
        k = WedgeKernel(**_simple_primitives())
        pts = tf.constant([[0.5, 0.0, 3.0]], dtype=tf.float64)
        Z = k._embed(pts)
        npt.assert_allclose(Z[0, 1:].numpy(), [0.0, 0.0], atol=1e-10)

    def test_incomparable_distance_depends_on_value(self) -> None:
        k = WedgeKernel(**_simple_primitives())
        inactive = tf.constant([[0.5, 0.0, 0.0]], dtype=tf.float64)
        active_low = tf.constant([[0.5, 1.0, 0.5]], dtype=tf.float64)
        active_high = tf.constant([[0.5, 1.0, 4.5]], dtype=tf.float64)
        dist_low = tf.norm(k._embed(inactive) - k._embed(active_low)).numpy()
        dist_high = tf.norm(k._embed(inactive) - k._embed(active_high)).numpy()
        assert dist_low != pytest.approx(dist_high, abs=1e-4)

    def test_wedge_differs_from_arc_incomparable(self) -> None:
        k_arc = ArcKernel(**_simple_primitives())
        k_wedge = WedgeKernel(**_simple_primitives())
        inactive = tf.constant([[0.5, 0.0, 0.0]], dtype=tf.float64)
        active = tf.constant([[0.5, 1.0, 2.5]], dtype=tf.float64)
        K_arc = k_arc.K(inactive, active).numpy().item()
        K_wedge = k_wedge.K(inactive, active).numpy().item()
        assert K_arc != pytest.approx(K_wedge, abs=1e-4)

    def test_active_embedding_scales_with_value(self) -> None:
        k = WedgeKernel(**_simple_primitives())
        low = tf.constant([[0.5, 1.0, 1.0]], dtype=tf.float64)
        high = tf.constant([[0.5, 1.0, 4.0]], dtype=tf.float64)
        mag_low = tf.norm(k._embed(low)[0, 1:]).numpy()
        mag_high = tf.norm(k._embed(high)[0, 1:]).numpy()
        assert mag_high > mag_low


# ---------------------------------------------------------------------------
# Basic GPR integration (still pure-gpflow: no trieste.space imports)
# ---------------------------------------------------------------------------


class TestKernelGPRIntegration:
    @pytest.mark.parametrize("kernel_cls_", [ArcKernel, WedgeKernel])
    def test_gpr_predict_valid(self, kernel_cls_) -> None:
        kernel = gpflow.kernels.Constant() * kernel_cls_(**_simple_primitives())

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
            data=(X_train, Y_train), kernel=kernel, noise_variance=0.1
        )

        X_test = np.array([[0.3, 1.0, 2.0], [0.5, 0.0, 1.5]])
        mean, var = gpr.predict_f(X_test)

        assert mean.shape == (2, 1)
        assert var.shape == (2, 1)
        assert tf.reduce_all(var > 0.0).numpy()
        assert tf.reduce_all(tf.math.is_finite(mean)).numpy()
        assert tf.reduce_all(tf.math.is_finite(var)).numpy()

    @pytest.mark.parametrize("kernel_cls_", [ArcKernel, WedgeKernel])
    def test_gpr_log_marginal_likelihood_finite(self, kernel_cls_) -> None:
        kernel = gpflow.kernels.Constant() * kernel_cls_(**_simple_primitives())
        X = np.array([[0.2, 1.0, 1.0], [0.4, 1.0, 3.0], [0.6, 0.0, 2.0]])
        Y = np.array([[0.1], [0.5], [-0.3]])
        gpr = gpflow.models.GPR(data=(X, Y), kernel=kernel, noise_variance=0.1)
        assert tf.math.is_finite(gpr.log_marginal_likelihood()).numpy()

    @pytest.mark.parametrize("kernel_cls_", [ArcKernel, WedgeKernel])
    def test_two_indicator_k_is_psd(self, kernel_cls_) -> None:
        k = kernel_cls_(**_two_indicator_primitives())
        pts = tf.constant(
            [
                [0.5, 1.0, 1.0, 3.0, 0.5],  # both active
                [0.5, 1.0, 0.0, 3.0, 0.5],  # x2 active, x3 inactive
                [0.5, 0.0, 1.0, 3.0, 0.5],  # x2 inactive, x3 active
                [0.5, 0.0, 0.0, 3.0, 0.5],  # both inactive
            ],
            dtype=tf.float64,
        )
        K = k.K(pts, pts)
        assert K.shape == (4, 4)
        eigvals = tf.linalg.eigvalsh(K)
        assert tf.reduce_all(eigvals >= -1e-6).numpy()
