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
"""Tests for the hierarchical-search-space API."""
from __future__ import annotations

import gpflow.kernels
import numpy.testing as npt
import pytest
import tensorflow as tf

from trieste.space import (
    INACTIVE_CONSTRAINT_RESIDUAL,
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
)

# ===== Worked-example fixture =====


def _worked_example_subspaces() -> dict[str, SearchSpace]:
    """Five-variable worked example.

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
        HierarchyNode(
            "shared",
            subspace_tags=["x1"],
        ),
        HierarchyNode(
            "branch_A",
            subspace_tags=["x2", "x4"],
            activity_condition_tags={"y1": 1},
        ),
        HierarchyNode(
            "branch_B",
            subspace_tags=["x3"],
            activity_condition_tags={"y1": 0},
        ),
    ]


def _make_worked_example_hss() -> HierarchicalSearchSpace:
    subspaces = _worked_example_subspaces()
    return HierarchicalSearchSpace(
        subspaces,
        _worked_example_hierarchy(subspaces),
    )


# ===== tag resolution (via to_gpflow_hierarchy) =====


def test_resolves_subspace_tags_to_columns() -> None:
    subspaces = _worked_example_subspaces()
    space = HierarchicalSearchSpace(
        subspaces,
        [
            HierarchyNode("shared", subspace_tags=["x1"]),
            HierarchyNode(
                "branch_A",
                subspace_tags=["x2", "x4"],
                activity_condition_tags={"y1": 1},
            ),
            HierarchyNode(
                "branch_B",
                subspace_tags=["x3"],
                activity_condition_tags={"y1": 0},
            ),
        ],
    )
    by_name = {n.name: n for n in space.to_gpflow_hierarchy()}
    # x1=col0, y1=col1, x2=col2, x4=col3, x3=col4
    assert list(by_name["branch_A"].feature_dims) == [2, 3]


def test_translates_activity_condition_tags_to_global_columns() -> None:
    subspaces = _worked_example_subspaces()
    space = _make_worked_example_hss()
    by_name = {n.name: n for n in space.to_gpflow_hierarchy()}
    # y1 is at flat-vector column 1 (x1=0, y1=1, ...), and that column is the key.
    assert by_name["branch_A"].activity_condition.requirements == {1: 1}


def test_preserves_categorical_int_value() -> None:
    subspaces = {
        "x1": Box([0.0], [1.0]),
        "y1": CategoricalSearchSpace(3),
        "x2": Box([0.0], [5.0]),
    }
    space = HierarchicalSearchSpace(
        subspaces,
        [
            HierarchyNode("shared", subspace_tags=["x1"]),
            HierarchyNode(
                "branch",
                subspace_tags=["x2"],
                activity_condition_tags={"y1": 2},
            ),
        ],
    )
    by_name = {n.name: n for n in space.to_gpflow_hierarchy()}
    # Value 2 must survive as int (not bool-coerced to True). y1 is at column 1.
    req = by_name["branch"].activity_condition.requirements
    assert req[1] == 2
    assert not isinstance(req[1], bool)


def test_stacks_feature_bounds() -> None:
    space = _make_worked_example_hss()
    by_name = {n.name: n for n in space.to_gpflow_hierarchy()}
    bounds = tf.convert_to_tensor(by_name["branch_A"].feature_bounds, dtype=tf.float64).numpy()
    npt.assert_array_almost_equal(bounds, [[0.0, 5.0], [-2.0, 2.0]])


def test_default_activity_condition_is_empty() -> None:
    space = _make_worked_example_hss()
    by_name = {n.name: n for n in space.to_gpflow_hierarchy()}
    assert by_name["shared"].activity_condition.requirements == {}


def test_expands_multidimensional_subspace_to_one_col_per_dim() -> None:
    """A non-indicator subspace of dimension d contributes d consecutive columns."""
    subspaces = {
        "x1": Box([0.0, 0.0], [1.0, 1.0]),  # 2-D
        "y1": BooleanSearchSpace(),
        "x2": Box([0.0], [1.0]),
    }
    space = HierarchicalSearchSpace(
        subspaces,
        [
            HierarchyNode("shared", subspace_tags=["x1"]),
            HierarchyNode(
                "branch",
                subspace_tags=["x2"],
                activity_condition_tags={"y1": 1},
            ),
        ],
    )
    by_name = {n.name: n for n in space.to_gpflow_hierarchy()}
    assert list(by_name["shared"].feature_dims) == [0, 1]


def test_infers_indicator_when_indicator_tags_omitted() -> None:
    # The activity_condition_tags key self-identifies as an indicator and is resolved to its
    # global column (y1 -> column 1) from subspaces alone.
    space = _make_worked_example_hss()
    by_name = {n.name: n for n in space.to_gpflow_hierarchy()}
    assert by_name["branch_A"].activity_condition.requirements == {1: 1}


def test_rejects_activity_condition_key_not_a_subspace() -> None:
    subspaces = _worked_example_subspaces()
    node = HierarchyNode(
        "branch_A",
        subspace_tags=["x2"],
        activity_condition_tags={"nope": 1},
    )
    with pytest.raises(ValueError, match="is not a key of"):
        HierarchicalSearchSpace(subspaces, [node])


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


def test_hss_multi_indicator_construction() -> None:
    # Two indicators: columns are [x1, y1, y2, x2].
    subspaces = {
        "x1": Box([0.0], [1.0]),
        "y1": BooleanSearchSpace(),
        "y2": BooleanSearchSpace(),
        "x2": Box([0.0], [1.0]),
    }
    hierarchy = [
        HierarchyNode("shared", subspace_tags=["x1"]),
        HierarchyNode(
            "branch",
            subspace_tags=["x2"],
            activity_condition_tags={"y1": 1, "y2": 0},
        ),
    ]
    space = HierarchicalSearchSpace(subspaces, hierarchy)
    assert space.indicator_tags == ("y1", "y2")
    assert list(space.indicator_dims) == [1, 2]
    assert space.non_indicator_tags == ("x1", "x2")


def test_hss_infers_indicator_tags_when_omitted() -> None:
    # Omitting indicator_tags must reproduce the explicit result on the worked example.
    subspaces = _worked_example_subspaces()
    inferred = HierarchicalSearchSpace(subspaces, _worked_example_hierarchy(subspaces))
    explicit = _make_worked_example_hss()
    assert inferred.indicator_tags == explicit.indicator_tags == ("y1",)
    assert list(inferred.indicator_dims) == list(explicit.indicator_dims) == [1]
    assert inferred.non_indicator_tags == explicit.non_indicator_tags


def test_hss_non_gated_boolean_inferred_as_feature() -> None:
    # A Boolean that no node gates on is, by role, a plain feature (not an indicator).
    subspaces = {
        "x1": Box([0.0], [1.0]),
        "b": BooleanSearchSpace(),  # used as a feature, never gates
        "y1": BooleanSearchSpace(),  # the actual indicator
        "x2": Box([0.0], [1.0]),
    }
    hierarchy = [
        HierarchyNode("shared", subspace_tags=["x1", "b"]),
        HierarchyNode("branch", subspace_tags=["x2"], activity_condition_tags={"y1": 1}),
    ]
    space = HierarchicalSearchSpace(subspaces, hierarchy)  # indicator_tags inferred
    assert space.indicator_tags == ("y1",)
    assert "b" in space.non_indicator_tags


def test_hss_inferred_box_indicator_rejected_by_type_check() -> None:
    # If a node gates on a Box column, inference treats it as an indicator and the type check
    # rejects it (indicators must be Boolean / 1-D categorical).
    subspaces = {"x1": Box([0.0], [1.0]), "x2": Box([0.0], [1.0])}
    hierarchy = [
        HierarchyNode("n", subspace_tags=["x1"], activity_condition_tags={"x2": 1})
    ]
    with pytest.raises(ValueError, match="BooleanSearchSpace or a"):
        HierarchicalSearchSpace(subspaces, hierarchy)


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
    """Boolean indicator values are accepted and compared as 1/0, since ``bool`` is a
    subtype of ``int`` in Python (``True == 1``, ``False == 0``)."""
    space = _make_worked_example_hss()
    assert set(space.active_subspace_tags({"y1": True})) == {"x1", "x2", "x4"}
    assert set(space.active_subspace_tags({"y1": False})) == {"x1", "x3"}


def test_hss_enumerate_tasks_boolean() -> None:
    space = _make_worked_example_hss()
    tasks = space.enumerate_tasks()
    assert len(tasks) == 2
    assert {"y1": 0} in tasks
    assert {"y1": 1} in tasks


def test_hss_enumerate_tasks_no_indicators() -> None:
    # A space with no indicators has a single (empty) task.
    subspaces = {"x1": Box([0.0], [1.0])}
    hierarchy = [HierarchyNode("only", subspace_tags=["x1"])]
    space = HierarchicalSearchSpace(subspaces, hierarchy)
    assert space.enumerate_tasks() == [{}]


def test_hss_enumerate_tasks_two_indicators() -> None:
    subspaces = {
        "x1": Box([0.0], [1.0]),
        "y1": BooleanSearchSpace(),
        "y2": BooleanSearchSpace(),
        "x2": Box([0.0], [1.0]),
        "x3": Box([0.0], [1.0]),
    }
    hierarchy = [
        HierarchyNode("shared", subspace_tags=["x1"]),
        HierarchyNode("a", subspace_tags=["x2"], activity_condition_tags={"y1": 1}),
        HierarchyNode("b", subspace_tags=["x3"], activity_condition_tags={"y2": 1}),
    ]
    space = HierarchicalSearchSpace(subspaces, hierarchy)
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


def test_hss_active_subspace_tags_requires_complete_config() -> None:
    # A partial config (an indicator omitted) is rejected rather than silently treated as off.
    space = _make_worked_example_hss()  # one indicator: y1
    with pytest.raises(ValueError, match="every indicator"):
        space.active_subspace_tags({})
    with pytest.raises(ValueError, match="every indicator"):
        space.is_active("x2", {})


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


def test_hss_raises_if_indicator_tag_refs_non_indicator_subspace() -> None:
    # A node gating on a DiscreteSearchSpace makes it an inferred indicator, but a discrete
    # space is neither Boolean nor a 1-D categorical, so construction must fail.
    subspaces = {
        "x1": Box([0.0], [1.0]),
        "y1": DiscreteSearchSpace(tf.constant([[0], [1], [2]])),
    }
    hierarchy = [
        HierarchyNode("n", subspace_tags=["x1"], activity_condition_tags={"y1": 1})
    ]
    with pytest.raises(ValueError, match="BooleanSearchSpace"):
        HierarchicalSearchSpace(subspaces, hierarchy)


def test_hss_raises_if_required_value_not_in_permitted_set() -> None:
    # y1 is a 3-ary categorical (permitted {0, 1, 2}); requiring value 5 is out of range.
    subspaces = {
        "x1": Box([0.0], [1.0]),
        "y1": CategoricalSearchSpace(3),
        "x2": Box([0.0], [1.0]),
    }
    hierarchy = [
        HierarchyNode("shared", subspace_tags=["x1"]),
        HierarchyNode("branch", subspace_tags=["x2"], activity_condition_tags={"y1": 5}),
    ]
    with pytest.raises(ValueError, match="permitted set"):
        HierarchicalSearchSpace(subspaces, hierarchy)


def test_hss_raises_if_orphan_non_indicator_column() -> None:
    subspaces = {
        "x1": Box([0.0], [1.0]),
        "x2": Box([0.0], [1.0]),
        "y1": BooleanSearchSpace(),
    }
    # x2 is never referenced by any node's subspace_tags.
    hierarchy = [
        HierarchyNode("n", subspace_tags=["x1"], activity_condition_tags={"y1": 1})
    ]
    with pytest.raises(ValueError, match="orphan"):
        HierarchicalSearchSpace(subspaces, hierarchy)


# ===== Categorical-indicator variant =====


def _make_categorical_hss() -> HierarchicalSearchSpace:
    subspaces = {
        "x1": Box([0.0], [1.0]),
        "y1": CategoricalSearchSpace(3),
        "x2": Box([0.0], [5.0]),
        "x3": Box([-1.0], [1.0]),
    }
    hierarchy = [
        HierarchyNode("shared", subspace_tags=["x1"]),
        HierarchyNode("branch_A", subspace_tags=["x2"], activity_condition_tags={"y1": 1}),
        HierarchyNode("branch_B", subspace_tags=["x3"], activity_condition_tags={"y1": 2}),
    ]
    return HierarchicalSearchSpace(subspaces, hierarchy)


def test_categorical_hss_enumerate_tasks_three_configs() -> None:
    tasks = _make_categorical_hss().enumerate_tasks()
    assert tasks == [{"y1": 0}, {"y1": 1}, {"y1": 2}]


def test_categorical_hss_active_subspaces() -> None:
    space = _make_categorical_hss()
    assert space.active_subspace_tags({"y1": 0}) == ["x1"]
    assert set(space.active_subspace_tags({"y1": 1})) == {"x1", "x2"}
    assert set(space.active_subspace_tags({"y1": 2})) == {"x1", "x3"}


# ===== product =====


def _make_second_hss(**kwargs) -> HierarchicalSearchSpace:
    """A second, disjoint hierarchical space (tags z1/w1/z2) for product tests.

    Columns: z1 (uncond), w1 (Boolean indicator), z2 gated by w1=1. Extra keyword
    arguments (e.g. ``constraints=``) are forwarded to the constructor.
    """
    subspaces = {
        "z1": Box([0.0], [1.0]),
        "w1": BooleanSearchSpace(),
        "z2": Box([0.0], [2.0]),
    }
    hierarchy = [
        HierarchyNode("z_shared", subspace_tags=["z1"]),
        HierarchyNode("z_branch", subspace_tags=["z2"], activity_condition_tags={"w1": 1}),
    ]
    return HierarchicalSearchSpace(subspaces, hierarchy, **kwargs)


def test_hss_product_combines_tags_dimension_and_indicators() -> None:
    combined = _make_worked_example_hss().product(_make_second_hss())
    assert combined.subspace_tags == ("x1", "y1", "x2", "x4", "x3", "z1", "w1", "z2")
    assert int(combined.dimension) == 8
    assert combined.indicator_tags == ("y1", "w1")
    # y1 is column 1, w1 is column 6 in the combined flat vector.
    assert combined.indicator_dims == [1, 6]


def test_hss_product_is_order_dependent() -> None:
    # product concatenates self's layout then other's, so A x B != B x A.
    a, b = _make_worked_example_hss(), _make_second_hss()
    ab = a.product(b)
    ba = b.product(a)
    assert ab.subspace_tags == ("x1", "y1", "x2", "x4", "x3", "z1", "w1", "z2")
    assert ba.subspace_tags == ("z1", "w1", "z2", "x1", "y1", "x2", "x4", "x3")


def test_hss_product_shifts_other_feature_dims_and_requirements() -> None:
    combined = _make_worked_example_hss().product(_make_second_hss())
    # ``other``'s node feature_dims and activity_condition requirement columns are
    # both shifted by self.dimension (5): z2 -> column 7, and w1 (other column 1)
    # -> combined column 6.
    z_branch = next(node for node in combined.to_gpflow_hierarchy() if node.name == "z_branch")
    assert list(z_branch.feature_dims) == [7]
    assert dict(z_branch.activity_condition.requirements) == {6: 1}
    # feature_bounds describe values, not columns, so they are carried over unshifted.
    assert z_branch.feature_bounds.numpy().tolist() == [[0.0, 2.0]]


def test_hss_product_active_subspaces_and_is_active() -> None:
    combined = _make_worked_example_hss().product(_make_second_hss())
    assert set(combined.active_subspace_tags({"y1": 1, "w1": 1})) == {
        "x1",
        "x2",
        "x4",
        "z1",
        "z2",
    }
    assert set(combined.active_subspace_tags({"y1": 0, "w1": 0})) == {"x1", "x3", "z1"}
    assert combined.is_active("z2", {"y1": 0, "w1": 1})
    assert not combined.is_active("z2", {"y1": 0, "w1": 0})


def test_hss_product_raises_on_overlapping_tags() -> None:
    space = _make_worked_example_hss()
    with pytest.raises(ValueError, match="overlapping tags"):
        space.product(space)


# ===== additional validation =====


def test_hss_raises_if_non_indicator_subspace_has_no_bounds() -> None:
    # c1 is a non-indicator categorical subspace owned by a node: it has no numerical bounds,
    # so it cannot be encoded as a (lower, upper) feature_bounds row.
    subspaces = {
        "x1": Box([0.0], [1.0]),
        "c1": CategoricalSearchSpace(3),
        "y1": BooleanSearchSpace(),
    }
    hierarchy = [
        HierarchyNode(
            "n", subspace_tags=["x1", "c1"], activity_condition_tags={"y1": 1}
        )
    ]
    with pytest.raises(ValueError, match="without numerical"):
        HierarchicalSearchSpace(subspaces, hierarchy)


def test_raises_on_duplicate_subspace_tags() -> None:
    subspaces = {"x1": Box([0.0], [1.0]), "y1": BooleanSearchSpace()}
    node = HierarchyNode(
        "n", subspace_tags=["x1", "x1"], activity_condition_tags={"y1": 1}
    )
    with pytest.raises(ValueError, match="duplicate tags"):
        HierarchicalSearchSpace(subspaces, [node])


def test_rejects_empty_subspace_tags() -> None:
    subspaces = {"x1": Box([0.0], [1.0]), "y1": BooleanSearchSpace()}
    node = HierarchyNode("n", subspace_tags=[])
    with pytest.raises(ValueError, match="must be non-empty"):
        HierarchicalSearchSpace(subspaces, [node])


def test_rejects_subspace_tag_not_a_subspace() -> None:
    subspaces = {"x1": Box([0.0], [1.0]), "y1": BooleanSearchSpace()}
    node = HierarchyNode("n", subspace_tags=["missing"])
    with pytest.raises(ValueError, match="is not a key of"):
        HierarchicalSearchSpace(subspaces, [node])


def test_rejects_multidimensional_indicator() -> None:
    # An activity-condition key resolving to more than one column is not a valid indicator.
    subspaces = {"x1": Box([0.0], [1.0]), "big": Box([0.0, 0.0], [1.0, 1.0])}
    node = HierarchyNode(
        "n", subspace_tags=["x1"], activity_condition_tags={"big": 1}
    )
    with pytest.raises(ValueError, match="must be 1-dimensional"):
        HierarchicalSearchSpace(subspaces, [node])


def test_rejects_non_indicator_subspace_without_bounds() -> None:
    # A categorical (no numerical bounds) cannot be a non-indicator feature.
    subspaces = {
        "x1": Box([0.0], [1.0]),
        "c": CategoricalSearchSpace(3),
        "y1": BooleanSearchSpace(),
    }
    node = HierarchyNode(
        "n", subspace_tags=["c"], activity_condition_tags={"y1": 1}
    )
    with pytest.raises(ValueError, match="without numerical"):
        HierarchicalSearchSpace(subspaces, [node])


def test_accepts_discrete_non_indicator_subspace() -> None:
    # A discrete (bounded) subspace is allowed as a non-indicator feature.
    subspaces = {
        "x1": Box([0.0], [1.0]),
        "d": DiscreteSearchSpace(tf.constant([[0.0], [1.0], [2.0]], dtype=tf.float64)),
        "y1": BooleanSearchSpace(),
        "x2": Box([0.0], [1.0]),
    }
    space = HierarchicalSearchSpace(
        subspaces,
        [
            HierarchyNode("shared", subspace_tags=["x1", "d"]),
            HierarchyNode(
                "branch", subspace_tags=["x2"], activity_condition_tags={"y1": 1}
            ),
        ],
    )
    by_name = {n.name: n for n in space.to_gpflow_hierarchy()}
    # x1 -> col 0, d -> col 1; bounds reflect each subspace.
    assert list(by_name["shared"].feature_dims) == [0, 1]
    bounds = tf.convert_to_tensor(by_name["shared"].feature_bounds, dtype=tf.float64).numpy()
    npt.assert_array_almost_equal(bounds, [[0.0, 1.0], [0.0, 2.0]])


# ===== __eq__ / __repr__ =====


def test_hss_eq_true_for_identical_spaces() -> None:
    assert _make_worked_example_hss() == _make_worked_example_hss()


def test_hss_eq_false_when_hierarchy_differs() -> None:
    # Same subspaces, but a hierarchy whose node names differ.
    subspaces = _worked_example_subspaces()
    renamed_hierarchy = [
        HierarchyNode("shared_renamed", subspace_tags=["x1"]),
        HierarchyNode(
            "branch_A",
            subspace_tags=["x2", "x4"],
            activity_condition_tags={"y1": 1},
        ),
        HierarchyNode(
            "branch_B",
            subspace_tags=["x3"],
            activity_condition_tags={"y1": 0},
        ),
    ]
    other = HierarchicalSearchSpace(subspaces, renamed_hierarchy)
    assert _make_worked_example_hss() != other


def test_hss_eq_false_for_non_hierarchical_object() -> None:
    assert _make_worked_example_hss() != 5


def test_hss_repr_includes_hierarchy_and_indicator_tags() -> None:
    text = repr(_make_worked_example_hss())
    assert "hierarchy" in text
    assert "indicator_tags" in text
    assert "y1" in text
    assert "constraints" in text
    assert "ctol" in text


# ===== gpflow kernel integration =====


def test_hss_hierarchy_requirements_use_global_columns() -> None:
    space = _make_worked_example_hss()
    # Columns: [x1, y1, x2, x4, x3]; the single indicator y1 is at flat column 1,
    # and requirements are keyed by that global column (gpflow's convention).
    assert space.indicator_dims == [1]
    by_name = {n.name: n for n in space.to_gpflow_hierarchy()}
    assert dict(by_name["shared"].activity_condition.requirements) == {}
    assert dict(by_name["branch_A"].activity_condition.requirements) == {1: 1}
    assert dict(by_name["branch_B"].activity_condition.requirements) == {1: 0}


def test_hss_hierarchy_is_directly_consumable_by_arc_hierarchical() -> None:
    # No adapter needed: to_gpflow_hierarchy() uses gpflow's column convention,
    # so feature columns and indicator columns tile active_dims contiguously.
    space = _make_worked_example_hss()
    active_dims = list(range(int(space.dimension)))
    kernel = gpflow.kernels.ArcHierarchical(
        space.to_gpflow_hierarchy(), active_dims=active_dims
    )
    # Two points differing only in the indicator are placed apart by the kernel.
    x = tf.constant(
        [
            [0.5, 1.0, 2.5, 0.0, 0.0],  # y1 = 1
            [0.5, 0.0, 2.5, 0.0, 0.0],  # y1 = 0
        ],
        dtype=tf.float64,
    )
    cov = kernel.K(x).numpy()
    assert cov[0, 1] < cov[0, 0]


# ===== global constraints (standard SearchSpace contract) =====


def _make_constrained_hss(constraints, ctol: float = 1e-7) -> HierarchicalSearchSpace:
    subspaces = _worked_example_subspaces()
    return HierarchicalSearchSpace(
        subspaces,
        _worked_example_hierarchy(subspaces),
        constraints=constraints,
        ctol=ctol,
    )


def test_hss_has_no_constraints_by_default() -> None:
    space = _make_worked_example_hss()
    assert space.has_constraints is False
    assert list(space.constraints) == []
    # With no constraints every point is feasible, and residuals are unavailable.
    pts = tf.constant([[0.5, 1.0, 2.0, 0.0, 0.0]], dtype=tf.float64)
    npt.assert_array_equal(space.is_feasible(pts).numpy(), [True])
    with pytest.raises(NotImplementedError):
        space.constraints_residuals(pts)


def test_hss_global_linear_constraint_residuals_and_feasibility() -> None:
    # 0.3 <= x1 <= 0.8, where x1 is flat-vector column 0.
    A = tf.constant([[1.0, 0.0, 0.0, 0.0, 0.0]], dtype=tf.float64)
    space = _make_constrained_hss([LinearConstraint(A=A, lb=[0.3], ub=[0.8])])
    assert space.has_constraints is True
    pts = tf.constant(
        [
            [0.5, 1.0, 2.0, 0.0, 0.0],  # x1 = 0.5 -> feasible
            [0.1, 0.0, 2.0, 0.0, 0.0],  # x1 = 0.1 < 0.3 -> infeasible
            [0.9, 1.0, 2.0, 0.0, 0.0],  # x1 = 0.9 > 0.8 -> infeasible
        ],
        dtype=tf.float64,
    )
    residuals = space.constraints_residuals(pts).numpy()
    # residual = [x1 - lb, ub - x1]
    npt.assert_allclose(residuals, [[0.2, 0.3], [-0.2, 0.7], [0.6, -0.1]])
    npt.assert_array_equal(space.is_feasible(pts).numpy(), [True, False, False])


def test_hss_global_nonlinear_constraint_feasibility() -> None:
    # 0 <= x1**2 <= 0.5.
    space = _make_constrained_hss(
        [
            NonlinearConstraint(
                lambda x: tf.reduce_sum(x[..., :1] ** 2, axis=-1, keepdims=True),
                lb=0.0,
                ub=0.5,
            )
        ]
    )
    pts = tf.constant(
        [
            [0.5, 1.0, 2.0, 0.0, 0.0],  # 0.25 in [0, 0.5] -> feasible
            [0.9, 0.0, 2.0, 0.0, 0.0],  # 0.81 > 0.5 -> infeasible
        ],
        dtype=tf.float64,
    )
    npt.assert_array_equal(space.is_feasible(pts).numpy(), [True, False])


def test_hss_product_combines_global_linear_constraints() -> None:
    # self (dim 5, cols [x1,y1,x2,x4,x3]) constrains 0.3 <= x1 <= 0.8 (col 0).
    self_space = _make_constrained_hss(
        [LinearConstraint(A=tf.constant([[1.0, 0.0, 0.0, 0.0, 0.0]], dtype=tf.float64), lb=[0.3], ub=[0.8])]
    )
    # other (dim 3, cols [z1,w1,z2]) constrains 0.0 <= z1 <= 0.5 (its col 0 -> combined col 5).
    other = _make_second_hss(
        constraints=[LinearConstraint(A=tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float64), lb=[0.0], ub=[0.5])]
    )
    combined = self_space.product(other)
    assert int(combined.dimension) == 8
    assert len(combined.constraints) == 2

    # combined columns: [x1, y1, x2, x4, x3, z1, w1, z2]
    pts = tf.constant(
        [
            [0.5, 1.0, 2.0, 0.0, 0.0, 0.3, 1.0, 1.0],  # x1 & z1 ok -> feasible
            [0.1, 1.0, 2.0, 0.0, 0.0, 0.3, 1.0, 1.0],  # x1 = 0.1 < 0.3 -> infeasible
            [0.5, 1.0, 2.0, 0.0, 0.0, 0.7, 1.0, 1.0],  # z1 = 0.7 > 0.5 -> infeasible
        ],
        dtype=tf.float64,
    )
    assert combined.constraints_residuals(pts).shape == (3, 4)  # 2 linear constraints x [lo, hi]
    npt.assert_array_equal(combined.is_feasible(pts).numpy(), [True, False, False])


def test_hss_product_embeds_nonlinear_constraint_on_correct_block() -> None:
    # other constrains z1**2 <= 0.25 on its own column 0 (-> combined column 5).
    other = _make_second_hss(
        constraints=[
            NonlinearConstraint(
                lambda x: tf.reduce_sum(x[..., :1] ** 2, axis=-1, keepdims=True), lb=0.0, ub=0.25
            )
        ]
    )
    combined = _make_worked_example_hss().product(other)
    base = tf.constant([[0.5, 1.0, 2.0, 0.0, 0.0, 0.3, 1.0, 1.0]], dtype=tf.float64)  # z1=0.3
    bad_z1 = tf.constant([[0.5, 1.0, 2.0, 0.0, 0.0, 0.9, 1.0, 1.0]], dtype=tf.float64)  # z1=0.9
    moved_x1 = tf.constant([[0.9, 1.0, 4.0, 1.0, 0.5, 0.3, 0.0, 1.5]], dtype=tf.float64)  # z1 unchanged

    npt.assert_array_equal(combined.is_feasible(base).numpy(), [True])  # 0.09 in [0, 0.25]
    npt.assert_array_equal(combined.is_feasible(bad_z1).numpy(), [False])  # 0.81 > 0.25
    # The embedded constraint reads only the z1 block: changing x-columns leaves the residual fixed.
    npt.assert_allclose(
        combined.constraints_residuals(base).numpy(), combined.constraints_residuals(moved_x1).numpy()
    )


def test_hss_product_uses_min_ctol() -> None:
    # The combined space takes the tighter (minimum) tolerance, regardless of operand order.
    constrained = _make_constrained_hss(
        [LinearConstraint(A=tf.constant([[1.0, 0.0, 0.0, 0.0, 0.0]], dtype=tf.float64), lb=[0.3], ub=[0.8])],
        ctol=1e-3,
    )
    other = _make_second_hss()  # default ctol 1e-7 (tighter)
    assert constrained.product(other).ctol == 1e-7
    assert other.product(constrained).ctol == 1e-7


# ===== conditional (disjunctive) constraints =====


def _hss_with_conditional(cc: ConditionalConstraint) -> HierarchicalSearchSpace:
    subspaces = _worked_example_subspaces()
    return HierarchicalSearchSpace(
        subspaces,
        _worked_example_hierarchy(subspaces),
        conditional_constraints=[cc],
    )


def _x3_in_unit_when_y1_zero() -> ConditionalConstraint:
    # -0.5 <= x3 <= 1.0, enforced only when y1 == 0.
    return ConditionalConstraint(
        constraint=LinearConstraint(A=tf.constant([[1.0]], dtype=tf.float64), lb=[-0.5], ub=[1.0]),
        indicator_conditions={"y1": 0},
        active_subspace_tags=["x3"],
    )


def test_conditional_constraint_active_returns_real_residual() -> None:
    space = _hss_with_conditional(_x3_in_unit_when_y1_zero())
    # y1 = 0 (active), x3 = 0.0 -> residual = [x3 - (-0.5), 1.0 - x3] = [0.5, 1.0]
    pts = tf.constant([[0.5, 0.0, 2.0, 0.0, 0.0]], dtype=tf.float64)
    npt.assert_allclose(space.constraints_residuals(pts).numpy(), [[0.5, 1.0]])
    npt.assert_array_equal(space.is_feasible(pts).numpy(), [True])


def test_conditional_constraint_inactive_returns_big_m() -> None:
    space = _hss_with_conditional(_x3_in_unit_when_y1_zero())
    # y1 = 1 (inactive): x3 = -0.8 would violate, but the constraint does not apply.
    pts = tf.constant([[0.5, 1.0, 2.0, 0.0, -0.8]], dtype=tf.float64)
    residuals = space.constraints_residuals(pts).numpy()
    npt.assert_array_equal(residuals, [[INACTIVE_CONSTRAINT_RESIDUAL] * 2])
    npt.assert_array_equal(space.is_feasible(pts).numpy(), [True])


def test_conditional_constraint_mixed_batch_feasibility() -> None:
    space = _hss_with_conditional(_x3_in_unit_when_y1_zero())
    pts = tf.constant(
        [
            [0.5, 0.0, 2.0, 0.0, 0.0],   # y1=0 active, x3=0.0 -> feasible
            [0.5, 0.0, 2.0, 0.0, -0.8],  # y1=0 active, x3=-0.8 -> infeasible
            [0.5, 1.0, 2.0, 0.0, -0.8],  # y1=1 inactive -> feasible (big-M)
        ],
        dtype=tf.float64,
    )
    npt.assert_array_equal(space.is_feasible(pts).numpy(), [True, False, True])


def test_conditional_constraint_categorical_indicator_matches_only_target() -> None:
    subspaces = {
        "x1": Box([0.0], [1.0]),
        "y1": CategoricalSearchSpace(3),  # column 1
        "x2": Box([-1.0], [1.0]),  # column 2
    }
    hierarchy = [
        HierarchyNode("shared", subspace_tags=["x1"]),
        HierarchyNode("branch", subspace_tags=["x2"], activity_condition_tags={"y1": 2}),
    ]
    cc = ConditionalConstraint(
        constraint=LinearConstraint(A=tf.constant([[1.0]], dtype=tf.float64), lb=[0.0], ub=[1.0]),
        indicator_conditions={"y1": 2},
        active_subspace_tags=["x2"],
    )
    space = HierarchicalSearchSpace(subspaces, hierarchy, conditional_constraints=[cc])
    pts = tf.constant(
        [
            [0.5, 2.0, -0.8],  # y1=2 active, x2=-0.8 < 0 -> infeasible
            [0.5, 1.0, -0.8],  # y1=1 inactive -> feasible
        ],
        dtype=tf.float64,
    )
    npt.assert_array_equal(space.is_feasible(pts).numpy(), [False, True])


def test_conditional_constraint_nonlinear_inner() -> None:
    cc = ConditionalConstraint(
        constraint=NonlinearConstraint(
            lambda x: tf.reduce_sum(x ** 2, axis=-1, keepdims=True), lb=0.0, ub=0.25
        ),
        indicator_conditions={"y1": 0},
        active_subspace_tags=["x3"],
    )
    space = _hss_with_conditional(cc)
    pts = tf.constant(
        [
            [0.5, 0.0, 2.0, 0.0, 0.3],  # y1=0 active, 0.09 in [0,0.25] -> feasible
            [0.5, 0.0, 2.0, 0.0, 0.9],  # y1=0 active, 0.81 > 0.25 -> infeasible
        ],
        dtype=tf.float64,
    )
    npt.assert_array_equal(space.is_feasible(pts).numpy(), [True, False])


def test_hss_raises_if_conditional_constraint_unknown_indicator() -> None:
    cc = ConditionalConstraint(
        constraint=LinearConstraint(A=tf.constant([[1.0]], dtype=tf.float64), lb=[0.0], ub=[1.0]),
        indicator_conditions={"nope": 0},
        active_subspace_tags=["x3"],
    )
    with pytest.raises(ValueError, match="not a declared indicator"):
        _hss_with_conditional(cc)


def test_hss_raises_if_conditional_constraint_value_out_of_permitted_set() -> None:
    cc = ConditionalConstraint(
        constraint=LinearConstraint(A=tf.constant([[1.0]], dtype=tf.float64), lb=[0.0], ub=[1.0]),
        indicator_conditions={"y1": 5},  # Boolean indicator only permits {0, 1}
        active_subspace_tags=["x3"],
    )
    with pytest.raises(ValueError, match="permitted set"):
        _hss_with_conditional(cc)


def test_hss_raises_if_conditional_constraint_unknown_active_subspace_tag() -> None:
    cc = ConditionalConstraint(
        constraint=LinearConstraint(A=tf.constant([[1.0]], dtype=tf.float64), lb=[0.0], ub=[1.0]),
        indicator_conditions={"y1": 0},
        active_subspace_tags=["does_not_exist"],
    )
    with pytest.raises(ValueError, match="not a key of"):
        _hss_with_conditional(cc)


def test_hss_raises_if_conditional_constraint_active_subspace_tag_is_indicator() -> None:
    cc = ConditionalConstraint(
        constraint=LinearConstraint(A=tf.constant([[1.0]], dtype=tf.float64), lb=[0.0], ub=[1.0]),
        indicator_conditions={"y1": 0},
        active_subspace_tags=["y1"],  # an indicator
    )
    with pytest.raises(ValueError, match="is an indicator"):
        _hss_with_conditional(cc)


def test_hss_product_propagates_conditional_constraints() -> None:
    space = _hss_with_conditional(_x3_in_unit_when_y1_zero())
    combined = space.product(_make_second_hss())
    assert len(combined.conditional_constraints) == 1
    assert combined.has_constraints is True
    # The conditional still evaluates on the combined 8-D layout (tag references survive product):
    # combined columns are [x1=0, y1=1, x2=2, x4=3, x3=4, z1=5, w1=6, z2=7].
    active_pt = tf.constant([[0.5, 0.0, 2.0, 0.0, 0.0, 0.5, 0.0, 0.5]], dtype=tf.float64)
    npt.assert_allclose(combined.constraints_residuals(active_pt).numpy(), [[0.5, 1.0]])
    npt.assert_array_equal(combined.is_feasible(active_pt).numpy(), [True])
    inactive_pt = tf.constant([[0.5, 1.0, 2.0, 0.0, -0.8, 0.5, 0.0, 0.5]], dtype=tf.float64)
    npt.assert_array_equal(
        combined.constraints_residuals(inactive_pt).numpy(), [[INACTIVE_CONSTRAINT_RESIDUAL] * 2]
    )


def test_conditional_constraint_residuals_are_differentiable() -> None:
    # The big-M reformulation exists so gradient-based polish can flow through the residual; verify
    # constraints_residuals is differentiable w.r.t. x on the active branch (d/dx3 of [x3+0.5,
    # 1.0-x3] is [+1, -1]).
    space = _hss_with_conditional(_x3_in_unit_when_y1_zero())
    x = tf.Variable([[0.5, 0.0, 2.0, 0.0, 0.0]], dtype=tf.float64)  # y1 = 0 -> active
    with tf.GradientTape() as tape:
        residuals = space.constraints_residuals(x)  # shape [1, 2]
    jac = tape.jacobian(residuals, x)
    assert jac is not None
    npt.assert_allclose(jac.numpy()[0, :, 0, 4], [1.0, -1.0])
    # Inactive branch: finite big-M residual and a defined (zero) gradient, no NaN/None.
    x_inactive = tf.Variable([[0.5, 1.0, 2.0, 0.0, 0.0]], dtype=tf.float64)  # y1 = 1 -> inactive
    with tf.GradientTape() as tape:
        residuals = space.constraints_residuals(x_inactive)
    jac_inactive = tape.jacobian(residuals, x_inactive)
    assert jac_inactive is not None
    npt.assert_array_equal(jac_inactive.numpy()[0, :, 0, 4], [0.0, 0.0])


def _two_indicator_subspaces() -> dict[str, SearchSpace]:
    # Columns: [x1=0, y1=1, y2=2, x2=3]; y1, y2 Boolean indicators.
    return {
        "x1": Box([0.0], [1.0]),
        "y1": BooleanSearchSpace(),
        "y2": BooleanSearchSpace(),
        "x2": Box([-1.0], [1.0]),
    }


def _two_indicator_hierarchy(subspaces: dict[str, SearchSpace]) -> list[HierarchyNode]:
    # x2 is active (and both y1, y2 are thereby gated) only when y1 == 0 AND y2 == 1.
    return [
        HierarchyNode("shared", subspace_tags=["x1"]),
        HierarchyNode(
            "branch",
            subspace_tags=["x2"],
            activity_condition_tags={"y1": 0, "y2": 1},
        ),
    ]


def test_hss_is_active_multi_indicator_node() -> None:
    # A node gated on two indicators is active only when BOTH conditions hold (logical AND).
    subspaces = _two_indicator_subspaces()
    space = HierarchicalSearchSpace(
        subspaces, _two_indicator_hierarchy(subspaces)
    )
    assert space.is_active("x2", {"y1": 0, "y2": 1})
    assert not space.is_active("x2", {"y1": 0, "y2": 0})
    assert not space.is_active("x2", {"y1": 1, "y2": 1})
    assert set(space.active_subspace_tags({"y1": 0, "y2": 1})) == {"x1", "x2"}
    assert set(space.active_subspace_tags({"y1": 0, "y2": 0})) == {"x1"}


def test_conditional_constraint_multiple_indicator_conditions() -> None:
    # A conditional constraint gated on two indicators is active only when BOTH hold.
    subspaces = _two_indicator_subspaces()
    cc = ConditionalConstraint(
        constraint=LinearConstraint(A=tf.constant([[1.0]], dtype=tf.float64), lb=[0.0], ub=[1.0]),
        indicator_conditions={"y1": 0, "y2": 1},
        active_subspace_tags=["x2"],
    )
    space = HierarchicalSearchSpace(
        subspaces,
        _two_indicator_hierarchy(subspaces),
        conditional_constraints=[cc],
    )
    # columns: [x1=0, y1=1, y2=2, x2=3]; x2 < 0 violates 0 <= x2 <= 1 only when active.
    pts = tf.constant(
        [
            [0.5, 0.0, 1.0, -0.8],  # y1=0, y2=1 active, x2=-0.8 -> infeasible
            [0.5, 0.0, 0.0, -0.8],  # y2 != 1 -> inactive -> feasible
            [0.5, 1.0, 1.0, -0.8],  # y1 != 0 -> inactive -> feasible
        ],
        dtype=tf.float64,
    )
    npt.assert_array_equal(space.is_feasible(pts).numpy(), [False, True, True])


# ===== logical propositions =====


def _implies_proposition() -> LogicalProposition:
    # "y2 = 1 implies y1 = 1", i.e. feasible unless (y2 == 1 and y1 == 0).
    return LogicalProposition(
        fun=lambda ind: tf.logical_or(
            tf.not_equal(ind["y2"][..., 0], 1.0), tf.equal(ind["y1"][..., 0], 1.0)
        ),
        name="y2_implies_y1",
    )


def _two_indicator_hss(**kwargs) -> HierarchicalSearchSpace:
    # Columns: x1(0), y1(1), y2(2), x2(3).
    subspaces = {
        "x1": Box([0.0], [1.0]),
        "y1": BooleanSearchSpace(),
        "y2": BooleanSearchSpace(),
        "x2": Box([0.0], [1.0]),
    }
    hierarchy = [
        HierarchyNode("shared", subspace_tags=["x1"]),
        HierarchyNode(
            "branch", subspace_tags=["x2"], activity_condition_tags={"y1": 1, "y2": 1}
        ),
    ]
    return HierarchicalSearchSpace(subspaces, hierarchy, **kwargs)


def test_logical_proposition_filters_violating_points() -> None:
    space = _two_indicator_hss(logical_propositions=[_implies_proposition()])
    pts = tf.constant(
        [
            [0.5, 1.0, 1.0, 0.5],  # y2=1, y1=1 -> ok
            [0.5, 0.0, 1.0, 0.5],  # y2=1, y1=0 -> violates implication
            [0.5, 0.0, 0.0, 0.5],  # y2=0 -> ok
        ],
        dtype=tf.float64,
    )
    npt.assert_array_equal(space.is_feasible(pts).numpy(), [True, False, True])


def test_logical_proposition_only_has_constraints_but_no_residuals() -> None:
    space = _two_indicator_hss(logical_propositions=[_implies_proposition()])
    assert space.has_constraints is True
    # No gradient-compatible constraints -> residuals are unavailable.
    pts = tf.constant([[0.5, 1.0, 1.0, 0.5]], dtype=tf.float64)
    with pytest.raises(NotImplementedError):
        space.constraints_residuals(pts)


def test_constraints_residuals_excludes_logical_propositions() -> None:
    # A global constraint plus a logical proposition: residuals reflect only the global one.
    A = tf.constant([[1.0, 0.0, 0.0, 0.0]], dtype=tf.float64)
    space = _two_indicator_hss(
        constraints=[LinearConstraint(A=A, lb=[0.3], ub=[0.8])],
        logical_propositions=[_implies_proposition()],
    )
    pts = tf.constant([[0.5, 0.0, 1.0, 0.5]], dtype=tf.float64)  # global ok, proposition violated
    # residual columns come only from the global constraint (shape [N, 2]).
    assert space.constraints_residuals(pts).shape == (1, 2)
    # is_feasible still folds the proposition in and rejects the point.
    npt.assert_array_equal(space.is_feasible(pts).numpy(), [False])


def test_is_feasible_conjunction_across_all_sources() -> None:
    A = tf.constant([[1.0, 0.0, 0.0, 0.0]], dtype=tf.float64)
    space = _two_indicator_hss(
        constraints=[LinearConstraint(A=A, lb=[0.3], ub=[0.8])],
        conditional_constraints=[
            ConditionalConstraint(
                constraint=LinearConstraint(
                    A=tf.constant([[1.0]], dtype=tf.float64), lb=[0.0], ub=[0.6]
                ),
                indicator_conditions={"y1": 1, "y2": 1},
                active_subspace_tags=["x2"],
            )
        ],
        logical_propositions=[_implies_proposition()],
    )
    pts = tf.constant(
        [
            [0.5, 1.0, 1.0, 0.5],  # global ok, conditional active & ok, proposition ok -> True
            [0.1, 1.0, 1.0, 0.5],  # global x1=0.1<0.3 -> False
            [0.5, 1.0, 1.0, 0.9],  # conditional active, x2=0.9>0.6 -> False
            [0.5, 0.0, 1.0, 0.5],  # proposition violated (y2=1,y1=0) -> False
        ],
        dtype=tf.float64,
    )
    npt.assert_array_equal(space.is_feasible(pts).numpy(), [True, False, False, False])


def test_hss_product_propagates_logical_propositions() -> None:
    space = _two_indicator_hss(logical_propositions=[_implies_proposition()])
    other = _make_second_hss()  # disjoint tags z1/w1/z2
    combined = space.product(other)
    assert len(combined.logical_propositions) == 1
    assert combined.has_constraints is True


def test_hss_enumerate_tasks_feasible_only_filters_propositions() -> None:
    # "y2 = 1 implies y1 = 1" makes {"y1": 0, "y2": 1} infeasible.
    space = _two_indicator_hss(logical_propositions=[_implies_proposition()])
    assert len(space.enumerate_tasks()) == 4  # default: full product, propositions ignored
    feasible = space.enumerate_tasks(feasible_only=True)
    assert {"y1": 0, "y2": 1} not in feasible
    assert len(feasible) == 3
    assert all(not (t["y2"] == 1 and t["y1"] == 0) for t in feasible)


def test_hss_product_combines_all_constraint_sources() -> None:
    # self (cols [x1, y1, y2, x2]) carries a global, a conditional, and a logical constraint.
    self_space = _two_indicator_hss(
        constraints=[
            LinearConstraint(A=tf.constant([[1.0, 0.0, 0.0, 0.0]], dtype=tf.float64), lb=[0.0], ub=[0.8])
        ],
        conditional_constraints=[
            ConditionalConstraint(
                constraint=LinearConstraint(
                    A=tf.constant([[1.0]], dtype=tf.float64), lb=[0.0], ub=[0.6]
                ),
                indicator_conditions={"y1": 1, "y2": 1},
                active_subspace_tags=["x2"],
            )
        ],
        logical_propositions=[_implies_proposition()],
    )
    # other (cols [z1, w1, z2]) carries a global constraint (z1 -> combined col 4).
    other = _make_second_hss(
        constraints=[
            LinearConstraint(A=tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float64), lb=[0.0], ub=[0.5])
        ]
    )
    combined = self_space.product(other)

    # combined columns: [x1, y1, y2, x2, z1, w1, z2]
    assert int(combined.dimension) == 7
    assert len(combined.constraints) == 2  # self x1 + other z1, embedded into the wide vector
    assert len(combined.conditional_constraints) == 1
    assert len(combined.logical_propositions) == 1

    pts = tf.constant(
        [
            [0.5, 1.0, 1.0, 0.5, 0.3, 1.0, 1.0],  # every source satisfied -> feasible
            [0.5, 0.0, 1.0, 0.5, 0.3, 1.0, 1.0],  # logical violated (y2=1, y1=0) -> infeasible
            [0.5, 1.0, 1.0, 0.5, 0.7, 1.0, 1.0],  # other global z1=0.7 > 0.5 -> infeasible
            [0.5, 1.0, 1.0, 0.9, 0.3, 1.0, 1.0],  # conditional active, x2=0.9 > 0.6 -> infeasible
        ],
        dtype=tf.float64,
    )
    npt.assert_array_equal(combined.is_feasible(pts).numpy(), [True, False, False, False])
