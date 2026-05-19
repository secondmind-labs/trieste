# Copyright 2026 The Trieste Contributors
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
"""Tests for the PR-1 hierarchical-search-space API.

Covers the GPflow re-exports (``HierarchyNode``/``ActivityCondition``),
the ``hierarchy_node_from_tags`` tag-based builder, and the
``HierarchicalSearchSpace`` constructor that consumes a
``Mapping[str, SearchSpace]`` plus a sequence of
``gpflow.kernels.HierarchyNode``.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import gpflow.kernels
import numpy as np
import numpy.testing as npt
import pytest
import tensorflow as tf

from trieste.acquisition import FastConstraintsFeasibility
from trieste.space import (
    INACTIVE_CONSTRAINT_RESIDUAL,
    ActivityCondition,
    BooleanSearchSpace,
    Box,
    CategoricalSearchSpace,
    ConditionalConstraint,
    DiscreteSearchSpace,
    HierarchicalSearchSpace,
    HierarchyNode,
    LinearConstraint,
    LogicalProposition,
    NonlinearConstraint,
    SearchSpace,
    hierarchy_node_from_tags,
)


# ===== Re-export identity =====


def test_hierarchy_node_is_re_exported_from_gpflow() -> None:
    assert HierarchyNode is gpflow.kernels.HierarchyNode


def test_activity_condition_is_re_exported_from_gpflow() -> None:
    assert ActivityCondition is gpflow.kernels.ActivityCondition


# ===== Worked-example fixture =====


def _worked_example_subspaces() -> dict[str, SearchSpace]:
    """Five-variable worked example from the PR-1 design doc.

    Columns: x1 (uncond), y1 (Boolean indicator), x2 and x4 gated by y1=1,
    x3 gated by y1=0.
    """
    return {
        "x1": Box([0.0], [1.0]),
        "y1": BooleanSearchSpace(),
        "x2": Box([0.0], [5.0]),
        "x4": Box([-2.0], [2.0]),
        "x3": Box([-1.0], [1.0]),
    }


def _worked_example_hierarchy(subspaces: dict[str, SearchSpace]) -> list[HierarchyNode]:
    return [
        hierarchy_node_from_tags(
            "shared",
            subspace_tags=["x1"],
            subspaces=subspaces,
            indicator_tags=["y1"],
        ),
        hierarchy_node_from_tags(
            "branch_A",
            subspace_tags=["x2", "x4"],
            activity_condition_tags={"y1": 1},
            subspaces=subspaces,
            indicator_tags=["y1"],
        ),
        hierarchy_node_from_tags(
            "branch_B",
            subspace_tags=["x3"],
            activity_condition_tags={"y1": 0},
            subspaces=subspaces,
            indicator_tags=["y1"],
        ),
    ]


def _make_worked_example_hss() -> HierarchicalSearchSpace:
    subspaces = _worked_example_subspaces()
    return HierarchicalSearchSpace(
        subspaces,
        _worked_example_hierarchy(subspaces),
        indicator_tags=["y1"],
    )


# ===== hierarchy_node_from_tags =====


def test_helper_resolves_subspace_tags_to_columns() -> None:
    subspaces = _worked_example_subspaces()
    node = hierarchy_node_from_tags(
        "branch_A",
        subspace_tags=["x2", "x4"],
        activity_condition_tags={"y1": 1},
        subspaces=subspaces,
        indicator_tags=["y1"],
    )
    # x1=col0, y1=col1, x2=col2, x4=col3, x3=col4
    assert list(node.feature_dims) == [2, 3]


def test_helper_translates_activity_condition_tags_to_local_indices() -> None:
    subspaces = _worked_example_subspaces()
    node = hierarchy_node_from_tags(
        "branch_A",
        subspace_tags=["x2"],
        activity_condition_tags={"y1": 1},
        subspaces=subspaces,
        indicator_tags=["y1"],
    )
    assert node.activity_condition.requirements == {0: 1}


def test_helper_preserves_categorical_int_value() -> None:
    subspaces = {
        "x1": Box([0.0], [1.0]),
        "y1": CategoricalSearchSpace(3),
        "x2": Box([0.0], [5.0]),
    }
    node = hierarchy_node_from_tags(
        "branch",
        subspace_tags=["x2"],
        activity_condition_tags={"y1": 2},
        subspaces=subspaces,
        indicator_tags=["y1"],
    )
    # Value 2 must survive as int (not bool-coerced to True).
    req = node.activity_condition.requirements
    assert req[0] == 2
    assert not isinstance(req[0], bool)


def test_helper_stacks_feature_bounds() -> None:
    subspaces = _worked_example_subspaces()
    node = hierarchy_node_from_tags(
        "branch_A",
        subspace_tags=["x2", "x4"],
        activity_condition_tags={"y1": 1},
        subspaces=subspaces,
        indicator_tags=["y1"],
    )
    bounds = tf.convert_to_tensor(node.feature_bounds, dtype=tf.float64).numpy()
    npt.assert_array_almost_equal(bounds, [[0.0, 5.0], [-2.0, 2.0]])


def test_helper_default_activity_condition_is_empty() -> None:
    subspaces = _worked_example_subspaces()
    node = hierarchy_node_from_tags(
        "shared",
        subspace_tags=["x1"],
        subspaces=subspaces,
        indicator_tags=["y1"],
    )
    assert node.activity_condition.requirements == {}


def test_helper_expands_multidimensional_subspace_to_one_col_per_dim() -> None:
    """A non-indicator subspace of dimension d contributes d consecutive columns."""
    subspaces = {
        "x1": Box([0.0, 0.0], [1.0, 1.0]),  # 2-D
        "y1": BooleanSearchSpace(),
        "x2": Box([0.0], [1.0]),
    }
    node = hierarchy_node_from_tags(
        "shared",
        subspace_tags=["x1"],
        subspaces=subspaces,
        indicator_tags=["y1"],
    )
    assert list(node.feature_dims) == [0, 1]


# ===== HierarchicalSearchSpace: basic shape and properties =====


def test_hss_construction_valid() -> None:
    space = _make_worked_example_hss()
    assert int(space.dimension) == 5
    assert space.indicator_tags == ("y1",)
    assert space.non_indicator_tags == ("x1", "x2", "x4", "x3")
    assert len(space.hierarchy) == 3


def test_hss_indicator_dims() -> None:
    space = _make_worked_example_hss()
    assert list(space.indicator_dims) == [1]


def test_hss_lower_upper() -> None:
    space = _make_worked_example_hss()
    npt.assert_array_equal(space.lower, [0.0, 0.0, 0.0, -2.0, -1.0])
    npt.assert_array_equal(space.upper, [1.0, 1.0, 5.0, 2.0, 1.0])


def test_hss_sample_shape() -> None:
    space = _make_worked_example_hss()
    samples = space.sample(7)
    assert samples.shape == (7, 5)


def test_hss_sample_within_bounds() -> None:
    space = _make_worked_example_hss()
    for s in space.sample(20):
        assert s in space


def test_hss_contains() -> None:
    space = _make_worked_example_hss()
    # Columns: [x1, y1, x2, x4, x3]
    valid = tf.constant([0.5, 1.0, 2.0, 0.0, 0.0], dtype=tf.float64)
    invalid = tf.constant([0.5, 1.0, 6.0, 0.0, 0.0], dtype=tf.float64)
    assert valid in space
    assert invalid not in space


def test_hss_get_subspace_component() -> None:
    space = _make_worked_example_hss()
    point = tf.constant([[0.1, 1.0, 2.5, 1.5, -0.5]])
    npt.assert_array_almost_equal(space.get_subspace_component("x1", point), [[0.1]])
    npt.assert_array_almost_equal(space.get_subspace_component("y1", point), [[1.0]])
    npt.assert_array_almost_equal(space.get_subspace_component("x2", point), [[2.5]])
    npt.assert_array_almost_equal(space.get_subspace_component("x4", point), [[1.5]])
    npt.assert_array_almost_equal(space.get_subspace_component("x3", point), [[-0.5]])


# ===== Hierarchy queries =====


def test_hss_active_subspace_tags_y1_true() -> None:
    space = _make_worked_example_hss()
    active = space.active_subspace_tags({"y1": 1})
    assert set(active) == {"x1", "x2", "x4"}


def test_hss_active_subspace_tags_y1_false() -> None:
    space = _make_worked_example_hss()
    active = space.active_subspace_tags({"y1": 0})
    assert set(active) == {"x1", "x3"}


def test_hss_active_subspace_tags_with_bool_input() -> None:
    """Boolean inputs True/False are accepted as 1/0 for backward compatibility."""
    space = _make_worked_example_hss()
    assert set(space.active_subspace_tags({"y1": True})) == {"x1", "x2", "x4"}
    assert set(space.active_subspace_tags({"y1": False})) == {"x1", "x3"}


def test_hss_enumerate_tasks_boolean() -> None:
    space = _make_worked_example_hss()
    tasks = space.enumerate_tasks()
    assert len(tasks) == 2
    assert {"y1": 0} in tasks
    assert {"y1": 1} in tasks


def test_hss_enumerate_tasks_two_indicators() -> None:
    subspaces = {
        "x1": Box([0.0], [1.0]),
        "y1": BooleanSearchSpace(),
        "y2": BooleanSearchSpace(),
        "x2": Box([0.0], [1.0]),
        "x3": Box([0.0], [1.0]),
    }
    hierarchy = [
        hierarchy_node_from_tags(
            "shared", subspace_tags=["x1"],
            subspaces=subspaces, indicator_tags=["y1", "y2"],
        ),
        hierarchy_node_from_tags(
            "a", subspace_tags=["x2"], activity_condition_tags={"y1": 1},
            subspaces=subspaces, indicator_tags=["y1", "y2"],
        ),
        hierarchy_node_from_tags(
            "b", subspace_tags=["x3"], activity_condition_tags={"y2": 1},
            subspaces=subspaces, indicator_tags=["y1", "y2"],
        ),
    ]
    space = HierarchicalSearchSpace(subspaces, hierarchy, indicator_tags=["y1", "y2"])
    assert len(space.enumerate_tasks()) == 4


def test_hss_is_active() -> None:
    space = _make_worked_example_hss()
    assert space.is_active("x1", {"y1": 1})
    assert space.is_active("x1", {"y1": 0})
    assert space.is_active("x2", {"y1": 1})
    assert not space.is_active("x2", {"y1": 0})
    assert space.is_active("x4", {"y1": 1})
    assert not space.is_active("x4", {"y1": 0})
    assert space.is_active("x3", {"y1": 0})
    assert not space.is_active("x3", {"y1": 1})


def test_hss_node_for_subspace_shared_branch() -> None:
    space = _make_worked_example_hss()
    nodes_x1 = space.node_for_subspace("x1")
    assert [n.name for n in nodes_x1] == ["shared"]
    nodes_x2 = space.node_for_subspace("x2")
    assert [n.name for n in nodes_x2] == ["branch_A"]
    # branch_A also owns x4 — same node should come back.
    nodes_x4 = space.node_for_subspace("x4")
    assert [n.name for n in nodes_x4] == ["branch_A"]


# ===== Validation =====


def test_hss_raises_if_indicator_tag_not_a_key_of_subspaces() -> None:
    subspaces = {"x1": Box([0.0], [1.0]), "y1": BooleanSearchSpace()}
    hierarchy = [
        hierarchy_node_from_tags(
            "n", subspace_tags=["x1"], activity_condition_tags={"y1": 1},
            subspaces=subspaces, indicator_tags=["y1"],
        )
    ]
    with pytest.raises(ValueError, match="not a key of"):
        HierarchicalSearchSpace(subspaces, hierarchy, indicator_tags=["y_missing"])


def test_hss_raises_if_indicator_tag_refs_non_indicator_subspace() -> None:
    subspaces = {
        "x1": Box([0.0], [1.0]),
        "y1": DiscreteSearchSpace(tf.constant([[0], [1], [2]])),
    }
    # Using a "y1": 1 activity condition just to make hierarchy non-trivial; the
    # construction must fail because y1 isn't Boolean / 1-D Categorical.
    hierarchy = [
        gpflow.kernels.HierarchyNode(
            "n",
            feature_dims=[0],
            feature_bounds=tf.constant([[0.0, 1.0]], dtype=tf.float64),
            activity_condition=ActivityCondition({0: 1}),
        )
    ]
    with pytest.raises(ValueError, match="BooleanSearchSpace"):
        HierarchicalSearchSpace(subspaces, hierarchy, indicator_tags=["y1"])


def test_hss_raises_if_feature_dim_points_at_indicator_column() -> None:
    subspaces = {"x1": Box([0.0], [1.0]), "y1": BooleanSearchSpace()}
    # feature_dims=[1] points at y1 which is an indicator column.
    bad_node = gpflow.kernels.HierarchyNode(
        "n",
        feature_dims=[1],
        feature_bounds=tf.constant([[0.0, 1.0]], dtype=tf.float64),
        activity_condition=ActivityCondition({0: 1}),
    )
    with pytest.raises(ValueError, match="indicator column"):
        HierarchicalSearchSpace(subspaces, [bad_node], indicator_tags=["y1"])


def test_hss_raises_if_feature_dim_out_of_range() -> None:
    subspaces = {"x1": Box([0.0], [1.0]), "y1": BooleanSearchSpace()}
    bad_node = gpflow.kernels.HierarchyNode(
        "n",
        feature_dims=[42],
        feature_bounds=tf.constant([[0.0, 1.0]], dtype=tf.float64),
        activity_condition=ActivityCondition({0: 1}),
    )
    with pytest.raises(ValueError, match="out of range"):
        HierarchicalSearchSpace(subspaces, [bad_node], indicator_tags=["y1"])


def test_hss_raises_if_feature_bounds_disagree_with_subspace() -> None:
    subspaces = {"x1": Box([0.0], [1.0]), "y1": BooleanSearchSpace()}
    # x1 lives in [0, 1] but the node claims [-99, 99].
    bad_node = gpflow.kernels.HierarchyNode(
        "n",
        feature_dims=[0],
        feature_bounds=tf.constant([[-99.0, 99.0]], dtype=tf.float64),
        activity_condition=ActivityCondition({0: 1}),
    )
    with pytest.raises(ValueError, match="feature_bounds"):
        HierarchicalSearchSpace(subspaces, [bad_node], indicator_tags=["y1"])


def test_hss_raises_if_activity_condition_key_out_of_range() -> None:
    subspaces = {"x1": Box([0.0], [1.0]), "y1": BooleanSearchSpace()}
    # Only one indicator (local index 0), but the node references local index 7.
    bad_node = gpflow.kernels.HierarchyNode(
        "n",
        feature_dims=[0],
        feature_bounds=tf.constant([[0.0, 1.0]], dtype=tf.float64),
        activity_condition=ActivityCondition({7: 1}),
    )
    with pytest.raises(ValueError, match="indicator local index"):
        HierarchicalSearchSpace(subspaces, [bad_node], indicator_tags=["y1"])


def test_hss_raises_if_required_value_not_in_permitted_set() -> None:
    subspaces = {"x1": Box([0.0], [1.0]), "y1": CategoricalSearchSpace(3)}
    bad_node = gpflow.kernels.HierarchyNode(
        "n",
        feature_dims=[0],
        feature_bounds=tf.constant([[0.0, 1.0]], dtype=tf.float64),
        activity_condition=ActivityCondition({0: 5}),  # K=3, so 5 invalid
    )
    with pytest.raises(ValueError, match="permitted set"):
        HierarchicalSearchSpace(subspaces, [bad_node], indicator_tags=["y1"])


def test_hss_raises_if_orphan_non_indicator_column() -> None:
    subspaces = {
        "x1": Box([0.0], [1.0]),
        "x2": Box([0.0], [1.0]),
        "y1": BooleanSearchSpace(),
    }
    # x2 (column 1) is never referenced by any node's feature_dims.
    hierarchy = [
        hierarchy_node_from_tags(
            "n", subspace_tags=["x1"], activity_condition_tags={"y1": 1},
            subspaces=subspaces, indicator_tags=["y1"],
        )
    ]
    with pytest.raises(ValueError, match="orphan"):
        HierarchicalSearchSpace(subspaces, hierarchy, indicator_tags=["y1"])


def test_hss_raises_if_unused_indicator() -> None:
    subspaces = {
        "x1": Box([0.0], [1.0]),
        "y1": BooleanSearchSpace(),
        "y2": BooleanSearchSpace(),
    }
    hierarchy = [
        hierarchy_node_from_tags(
            "n", subspace_tags=["x1"], activity_condition_tags={"y1": 1},
            subspaces=subspaces, indicator_tags=["y1", "y2"],
        )
    ]
    with pytest.raises(ValueError, match="unused"):
        HierarchicalSearchSpace(subspaces, hierarchy, indicator_tags=["y1", "y2"])


# ===== Categorical-indicator variant =====


def _make_categorical_hss() -> HierarchicalSearchSpace:
    subspaces = {
        "x1": Box([0.0], [1.0]),
        "y1": CategoricalSearchSpace(3),
        "x2": Box([0.0], [5.0]),
        "x3": Box([-1.0], [1.0]),
    }
    hierarchy = [
        hierarchy_node_from_tags(
            "shared", subspace_tags=["x1"],
            subspaces=subspaces, indicator_tags=["y1"],
        ),
        hierarchy_node_from_tags(
            "branch_A", subspace_tags=["x2"], activity_condition_tags={"y1": 1},
            subspaces=subspaces, indicator_tags=["y1"],
        ),
        hierarchy_node_from_tags(
            "branch_B", subspace_tags=["x3"], activity_condition_tags={"y1": 2},
            subspaces=subspaces, indicator_tags=["y1"],
        ),
    ]
    return HierarchicalSearchSpace(subspaces, hierarchy, indicator_tags=["y1"])


def test_categorical_hss_enumerate_tasks_three_configs() -> None:
    tasks = _make_categorical_hss().enumerate_tasks()
    assert tasks == [{"y1": 0}, {"y1": 1}, {"y1": 2}]


def test_categorical_hss_active_subspaces() -> None:
    space = _make_categorical_hss()
    assert space.active_subspace_tags({"y1": 0}) == ["x1"]
    assert set(space.active_subspace_tags({"y1": 1})) == {"x1", "x2"}
    assert set(space.active_subspace_tags({"y1": 2})) == {"x1", "x3"}


# ===== Constraint machinery (PR-1b) =====
#
# Layout used throughout: subspaces = {x1, y1, x2, x4, x3}, so the flat-vector
# columns are x1=0, y1=1, x2=2, x4=3, x3=4. ``_worked_example_hierarchy`` gates
# x2/x4 by y1=1 and x3 by y1=0; ``_make_categorical_hss`` uses a 3-ary
# categorical y1.


def _x3_lower_bound_constraint() -> LinearConstraint:
    # A 1-D constraint enforcing x3 >= -0.5 (i.e. lb=-0.5, ub=+inf on A=[[1]]).
    return LinearConstraint(
        A=tf.constant([[1.0]], dtype=tf.float64),
        lb=tf.constant([-0.5], dtype=tf.float64),
        ub=tf.constant([np.inf], dtype=tf.float64),
    )


def _global_sum_constraint() -> LinearConstraint:
    # x1 + 0.5*x2 <= 0.9, zeros on y1, x4, x3.
    return LinearConstraint(
        A=tf.constant([[1.0, 0.0, 0.5, 0.0, 0.0]], dtype=tf.float64),
        lb=tf.constant([-np.inf], dtype=tf.float64),
        ub=tf.constant([0.9], dtype=tf.float64),
    )


# ----- ConditionalConstraint with a Boolean indicator -----


def test_conditional_constraint_boolean_indicator_residual_active() -> None:
    subspaces = _worked_example_subspaces()
    cc = ConditionalConstraint(
        constraint=_x3_lower_bound_constraint(),
        indicator_conditions={"y1": False},
        active_subspace_tags=["x3"],
    )
    space = HierarchicalSearchSpace(
        subspaces,
        _worked_example_hierarchy(subspaces),
        indicator_tags=["y1"],
        conditional_constraints=[cc],
    )
    # y1=0 → constraint active; x3=-0.2 ⇒ residual on lb is -0.2-(-0.5)=0.3 (feasible),
    # on ub is +inf-(-0.2)=+inf. x3=-0.8 ⇒ lb-residual -0.3 (violates).
    points = tf.constant(
        [
            [0.3, 0.0, 0.0, 0.0, -0.2],  # feasible (residual 0.3 on lb side)
            [0.3, 0.0, 0.0, 0.0, -0.8],  # infeasible (residual -0.3 on lb side)
        ],
        dtype=tf.float64,
    )
    residuals = cc.residual(points, space).numpy()
    npt.assert_allclose(residuals[0, 0], 0.3)
    npt.assert_allclose(residuals[1, 0], -0.3)


def test_conditional_constraint_boolean_indicator_residual_inactive() -> None:
    subspaces = _worked_example_subspaces()
    cc = ConditionalConstraint(
        constraint=_x3_lower_bound_constraint(),
        indicator_conditions={"y1": False},
        active_subspace_tags=["x3"],
    )
    space = HierarchicalSearchSpace(
        subspaces,
        _worked_example_hierarchy(subspaces),
        indicator_tags=["y1"],
        conditional_constraints=[cc],
    )
    # y1=1 ⇒ constraint inactive; residual columns should all be the Big-M sentinel.
    points = tf.constant([[0.3, 1.0, 1.5, 0.0, 0.0]], dtype=tf.float64)
    residuals = cc.residual(points, space).numpy()
    npt.assert_array_equal(residuals, np.full_like(residuals, INACTIVE_CONSTRAINT_RESIDUAL))


def test_conditional_constraint_mixed_batch_residual() -> None:
    subspaces = _worked_example_subspaces()
    cc = ConditionalConstraint(
        constraint=_x3_lower_bound_constraint(),
        indicator_conditions={"y1": False},
        active_subspace_tags=["x3"],
    )
    space = HierarchicalSearchSpace(
        subspaces,
        _worked_example_hierarchy(subspaces),
        indicator_tags=["y1"],
        conditional_constraints=[cc],
    )
    points = tf.constant(
        [
            [0.3, 1.0, 1.5, 0.0, 0.0],  # inactive → Big-M
            [0.3, 0.0, 0.0, 0.0, -0.2],  # active feasible
            [0.3, 0.0, 0.0, 0.0, -0.8],  # active infeasible
        ],
        dtype=tf.float64,
    )
    residuals = cc.residual(points, space).numpy()
    npt.assert_allclose(residuals[0, 0], INACTIVE_CONSTRAINT_RESIDUAL)
    npt.assert_allclose(residuals[1, 0], 0.3)
    npt.assert_allclose(residuals[2, 0], -0.3)


# ----- ConditionalConstraint with a K-ary categorical indicator -----


def _make_categorical_hss_with_conditional(
    indicator_conditions: Mapping[str, int],
) -> tuple[HierarchicalSearchSpace, ConditionalConstraint]:
    """Build the 3-ary categorical HSS with a conditional x3 >= -0.5 gated by ``indicator_conditions``."""
    subspaces = {
        "x1": Box([0.0], [1.0]),
        "y1": CategoricalSearchSpace(3),
        "x2": Box([0.0], [5.0]),
        "x3": Box([-1.0], [1.0]),
    }
    hierarchy = [
        hierarchy_node_from_tags(
            "shared", subspace_tags=["x1"], subspaces=subspaces, indicator_tags=["y1"]
        ),
        hierarchy_node_from_tags(
            "branch_A",
            subspace_tags=["x2"],
            activity_condition_tags={"y1": 1},
            subspaces=subspaces,
            indicator_tags=["y1"],
        ),
        hierarchy_node_from_tags(
            "branch_B",
            subspace_tags=["x3"],
            activity_condition_tags={"y1": 2},
            subspaces=subspaces,
            indicator_tags=["y1"],
        ),
    ]
    cc = ConditionalConstraint(
        constraint=_x3_lower_bound_constraint(),
        indicator_conditions=indicator_conditions,
        active_subspace_tags=["x3"],
    )
    space = HierarchicalSearchSpace(
        subspaces,
        hierarchy,
        indicator_tags=["y1"],
        conditional_constraints=[cc],
    )
    return space, cc


def test_conditional_constraint_categorical_indicator_matches_only_target_value() -> None:
    space, cc = _make_categorical_hss_with_conditional({"y1": 2})
    # Columns: x1, y1, x2, x3
    points = tf.constant(
        [
            [0.0, 0.0, 0.0, -0.2],  # y1=0 → inactive (Big-M)
            [0.0, 1.0, 0.0, -0.2],  # y1=1 → inactive (Big-M)
            [0.0, 2.0, 0.0, -0.2],  # y1=2 → active (residual = 0.3)
            [0.0, 2.0, 0.0, -0.8],  # y1=2 → active (residual = -0.3)
        ],
        dtype=tf.float64,
    )
    residuals = cc.residual(points, space).numpy()
    npt.assert_allclose(residuals[0, 0], INACTIVE_CONSTRAINT_RESIDUAL)
    npt.assert_allclose(residuals[1, 0], INACTIVE_CONSTRAINT_RESIDUAL)
    npt.assert_allclose(residuals[2, 0], 0.3)
    npt.assert_allclose(residuals[3, 0], -0.3)


# ----- ConditionalConstraint with a NonlinearConstraint inner -----


def test_conditional_constraint_with_nonlinear_inner() -> None:
    subspaces = _worked_example_subspaces()

    def fun(x: tf.Tensor) -> tf.Tensor:
        # x has shape [..., 1] (slice of x3); enforce x3**2 <= 0.25 (so -0.5 <= x3 <= 0.5).
        return tf.reduce_sum(x * x, axis=-1, keepdims=True)

    inner = NonlinearConstraint(
        fun=fun,
        lb=tf.constant([-np.inf], dtype=tf.float64),
        ub=tf.constant([0.25], dtype=tf.float64),
    )
    cc = ConditionalConstraint(
        constraint=inner,
        indicator_conditions={"y1": False},
        active_subspace_tags=["x3"],
    )
    space = HierarchicalSearchSpace(
        subspaces,
        _worked_example_hierarchy(subspaces),
        indicator_tags=["y1"],
        conditional_constraints=[cc],
    )
    # y1=0, x3=0.3 → x3**2=0.09 → ub_residual = 0.25-0.09 = 0.16 (feasible).
    # y1=0, x3=0.7 → x3**2=0.49 → ub_residual = 0.25-0.49 = -0.24 (infeasible).
    points = tf.constant(
        [
            [0.0, 0.0, 0.0, 0.0, 0.3],
            [0.0, 0.0, 0.0, 0.0, 0.7],
        ],
        dtype=tf.float64,
    )
    residuals = cc.residual(points, space).numpy()
    # NonlinearConstraint.residual returns [lb_residual, ub_residual] = 2 columns
    assert residuals.shape == (2, 2)
    npt.assert_allclose(residuals[0, 1], 0.16)
    npt.assert_allclose(residuals[1, 1], -0.24)


# ----- Constructor-time validation of conditional_constraints -----


def test_conditional_constraint_validation_unknown_indicator_tag() -> None:
    subspaces = _worked_example_subspaces()
    cc = ConditionalConstraint(
        constraint=_x3_lower_bound_constraint(),
        indicator_conditions={"nope": False},
        active_subspace_tags=["x3"],
    )
    with pytest.raises(ValueError, match="indicator_conditions key 'nope'"):
        HierarchicalSearchSpace(
            subspaces,
            _worked_example_hierarchy(subspaces),
            indicator_tags=["y1"],
            conditional_constraints=[cc],
        )


def test_conditional_constraint_validation_value_out_of_permitted_set_boolean() -> None:
    subspaces = _worked_example_subspaces()
    cc = ConditionalConstraint(
        constraint=_x3_lower_bound_constraint(),
        indicator_conditions={"y1": 5},  # 5 not in {0, 1}
        active_subspace_tags=["x3"],
    )
    with pytest.raises(ValueError, match="required value 5"):
        HierarchicalSearchSpace(
            subspaces,
            _worked_example_hierarchy(subspaces),
            indicator_tags=["y1"],
            conditional_constraints=[cc],
        )


def test_conditional_constraint_validation_value_out_of_permitted_set_categorical() -> None:
    subspaces = {
        "x1": Box([0.0], [1.0]),
        "y1": CategoricalSearchSpace(3),
        "x2": Box([0.0], [5.0]),
        "x3": Box([-1.0], [1.0]),
    }
    hierarchy = [
        hierarchy_node_from_tags(
            "shared", subspace_tags=["x1"], subspaces=subspaces, indicator_tags=["y1"]
        ),
        hierarchy_node_from_tags(
            "a", subspace_tags=["x2"], activity_condition_tags={"y1": 1},
            subspaces=subspaces, indicator_tags=["y1"],
        ),
        hierarchy_node_from_tags(
            "b", subspace_tags=["x3"], activity_condition_tags={"y1": 2},
            subspaces=subspaces, indicator_tags=["y1"],
        ),
    ]
    cc = ConditionalConstraint(
        constraint=_x3_lower_bound_constraint(),
        indicator_conditions={"y1": 3},  # 3 not in {0, 1, 2}
        active_subspace_tags=["x3"],
    )
    with pytest.raises(ValueError, match="required value 3"):
        HierarchicalSearchSpace(
            subspaces, hierarchy, indicator_tags=["y1"], conditional_constraints=[cc],
        )


def test_conditional_constraint_validation_unknown_active_subspace_tag() -> None:
    subspaces = _worked_example_subspaces()
    cc = ConditionalConstraint(
        constraint=_x3_lower_bound_constraint(),
        indicator_conditions={"y1": False},
        active_subspace_tags=["x99"],
    )
    with pytest.raises(ValueError, match="active_subspace_tags entry 'x99'"):
        HierarchicalSearchSpace(
            subspaces,
            _worked_example_hierarchy(subspaces),
            indicator_tags=["y1"],
            conditional_constraints=[cc],
        )


def test_conditional_constraint_validation_active_subspace_tag_is_indicator() -> None:
    subspaces = _worked_example_subspaces()
    cc = ConditionalConstraint(
        constraint=_x3_lower_bound_constraint(),
        indicator_conditions={"y1": False},
        active_subspace_tags=["y1"],  # indicator!
    )
    with pytest.raises(ValueError, match="refers to an indicator subspace"):
        HierarchicalSearchSpace(
            subspaces,
            _worked_example_hierarchy(subspaces),
            indicator_tags=["y1"],
            conditional_constraints=[cc],
        )


# ----- LogicalProposition -----


def _make_two_indicator_hss(
    *,
    logical_propositions: Sequence[LogicalProposition] = (),
) -> HierarchicalSearchSpace:
    subspaces = {
        "x1": Box([0.0], [1.0]),
        "y1": BooleanSearchSpace(),
        "y2": BooleanSearchSpace(),
        "x2": Box([0.0], [5.0]),
    }
    hierarchy = [
        hierarchy_node_from_tags(
            "shared", subspace_tags=["x1"], subspaces=subspaces, indicator_tags=["y1", "y2"]
        ),
        hierarchy_node_from_tags(
            "branch_A",
            subspace_tags=["x2"],
            activity_condition_tags={"y1": 1, "y2": 1},
            subspaces=subspaces,
            indicator_tags=["y1", "y2"],
        ),
    ]
    return HierarchicalSearchSpace(
        subspaces,
        hierarchy,
        indicator_tags=["y1", "y2"],
        logical_propositions=list(logical_propositions),
    )


def test_logical_proposition_filters_violating_points() -> None:
    # "if y2 = 1, then y1 = 1" — Boolean indicators.
    prop = LogicalProposition(
        fun=lambda ind: tf.logical_or(
            tf.equal(tf.cast(ind["y2"], tf.int32), 0),
            tf.equal(tf.cast(ind["y1"], tf.int32), 1),
        )[:, 0],
        name="y2_implies_y1",
    )
    space = _make_two_indicator_hss(logical_propositions=[prop])
    points = tf.constant(
        [
            [0.1, 0.0, 1.0, 0.0],  # y2=1, y1=0 → violates
            [0.1, 1.0, 1.0, 1.0],  # y2=1, y1=1 → satisfies
            [0.1, 0.0, 0.0, 0.0],  # y2=0 → vacuously True
        ],
        dtype=tf.float64,
    )
    feasible = space.is_feasible(points).numpy()
    npt.assert_array_equal(feasible, np.array([False, True, True]))


def test_logical_proposition_categorical_indicator() -> None:
    # "y1 = 2 is forbidden" on a 3-ary categorical.
    subspaces = {
        "x1": Box([0.0], [1.0]),
        "y1": CategoricalSearchSpace(3),
        "x2": Box([0.0], [5.0]),
        "x3": Box([-1.0], [1.0]),
    }
    hierarchy = [
        hierarchy_node_from_tags(
            "shared", subspace_tags=["x1"], subspaces=subspaces, indicator_tags=["y1"]
        ),
        hierarchy_node_from_tags(
            "a", subspace_tags=["x2"], activity_condition_tags={"y1": 1},
            subspaces=subspaces, indicator_tags=["y1"],
        ),
        hierarchy_node_from_tags(
            "b", subspace_tags=["x3"], activity_condition_tags={"y1": 2},
            subspaces=subspaces, indicator_tags=["y1"],
        ),
    ]
    prop = LogicalProposition(
        fun=lambda ind: tf.not_equal(tf.cast(ind["y1"], tf.int32), 2)[:, 0],
        name="no_category_two",
    )
    space = HierarchicalSearchSpace(
        subspaces, hierarchy, indicator_tags=["y1"], logical_propositions=[prop]
    )
    points = tf.constant(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
        ],
        dtype=tf.float64,
    )
    feasible = space.is_feasible(points).numpy()
    npt.assert_array_equal(feasible, np.array([True, True, False]))


# ----- has_constraints / constraints_residuals / is_feasible composition -----


def test_has_constraints_reflects_all_three_sources() -> None:
    subspaces = _worked_example_subspaces()
    hierarchy = _worked_example_hierarchy(subspaces)

    unconstrained = HierarchicalSearchSpace(subspaces, hierarchy, indicator_tags=["y1"])
    assert not unconstrained.has_constraints

    with_global = HierarchicalSearchSpace(
        subspaces,
        hierarchy,
        indicator_tags=["y1"],
        global_constraints=[_global_sum_constraint()],
    )
    assert with_global.has_constraints

    with_conditional = HierarchicalSearchSpace(
        subspaces,
        hierarchy,
        indicator_tags=["y1"],
        conditional_constraints=[
            ConditionalConstraint(
                constraint=_x3_lower_bound_constraint(),
                indicator_conditions={"y1": False},
                active_subspace_tags=["x3"],
            )
        ],
    )
    assert with_conditional.has_constraints

    with_proposition = HierarchicalSearchSpace(
        subspaces,
        hierarchy,
        indicator_tags=["y1"],
        logical_propositions=[
            LogicalProposition(fun=lambda ind: tf.ones([tf.shape(ind["y1"])[0]], dtype=tf.bool))
        ],
    )
    assert with_proposition.has_constraints


def test_constraints_residuals_excludes_logical_propositions() -> None:
    subspaces = _worked_example_subspaces()
    hierarchy = _worked_example_hierarchy(subspaces)
    cc = ConditionalConstraint(
        constraint=_x3_lower_bound_constraint(),
        indicator_conditions={"y1": False},
        active_subspace_tags=["x3"],
    )
    prop = LogicalProposition(
        fun=lambda ind: tf.zeros([tf.shape(ind["y1"])[0]], dtype=tf.bool),
        name="always_false",
    )
    space = HierarchicalSearchSpace(
        subspaces,
        hierarchy,
        indicator_tags=["y1"],
        global_constraints=[_global_sum_constraint()],
        conditional_constraints=[cc],
        logical_propositions=[prop],
    )
    points = tf.constant([[0.3, 0.0, 0.0, 0.0, -0.2]], dtype=tf.float64)
    residuals = space.constraints_residuals(points).numpy()
    # LinearConstraint.residual returns [lb, ub] per constraint row:
    # global (M=1) → 2 columns + conditional inner (M=1) → 2 columns = 4.
    assert residuals.shape == (1, 4)
    # is_feasible should be False because of the always-False proposition.
    assert bool(space.is_feasible(points).numpy()[0]) is False


def test_constraints_residuals_raises_when_only_proposition_present() -> None:
    space = _make_two_indicator_hss(
        logical_propositions=[
            LogicalProposition(fun=lambda ind: tf.ones([tf.shape(ind["y1"])[0]], dtype=tf.bool))
        ]
    )
    points = tf.constant([[0.1, 1.0, 1.0, 1.0]], dtype=tf.float64)
    with pytest.raises(NotImplementedError):
        space.constraints_residuals(points)


def test_is_feasible_takes_conjunction_across_all_sources() -> None:
    subspaces = _worked_example_subspaces()
    hierarchy = _worked_example_hierarchy(subspaces)
    cc = ConditionalConstraint(
        constraint=_x3_lower_bound_constraint(),
        indicator_conditions={"y1": False},
        active_subspace_tags=["x3"],
    )
    space = HierarchicalSearchSpace(
        subspaces,
        hierarchy,
        indicator_tags=["y1"],
        global_constraints=[_global_sum_constraint()],
        conditional_constraints=[cc],
    )
    points = tf.constant(
        [
            [0.3, 0.0, 0.0, 0.0, -0.2],  # global OK (0.3 <= 0.9), conditional OK
            [0.95, 0.0, 0.0, 0.0, -0.2],  # global VIOLATES (0.95 > 0.9)
            [0.3, 0.0, 0.0, 0.0, -0.8],  # conditional VIOLATES
            [0.3, 1.0, 1.0, 0.0, 0.0],  # y1=1: conditional inactive; global 0.3+0.5*1.0=0.8 OK
        ],
        dtype=tf.float64,
    )
    feasible = space.is_feasible(points).numpy()
    npt.assert_array_equal(feasible, np.array([True, False, False, True]))


# ----- FastConstraintsFeasibility smoke test -----


def test_fast_constraints_feasibility_consumes_constrained_hss() -> None:
    subspaces = _worked_example_subspaces()
    hierarchy = _worked_example_hierarchy(subspaces)
    cc = ConditionalConstraint(
        constraint=_x3_lower_bound_constraint(),
        indicator_conditions={"y1": False},
        active_subspace_tags=["x3"],
    )
    space = HierarchicalSearchSpace(
        subspaces,
        hierarchy,
        indicator_tags=["y1"],
        global_constraints=[_global_sum_constraint()],
        conditional_constraints=[cc],
    )
    builder = FastConstraintsFeasibility(space)
    acq = builder.prepare_acquisition_function(model=None)  # type: ignore[arg-type]
    points = tf.constant(
        [
            [0.3, 0.0, 0.0, 0.0, -0.2],
            [0.95, 0.0, 0.0, 0.0, -0.8],
        ],
        dtype=tf.float64,
    )
    values = acq(points).numpy()
    assert values.shape == (2,)
    # First point well inside the feasible region → high feasibility probability.
    # Second point violates both global and conditional → near-zero.
    assert values[0] > values[1]
