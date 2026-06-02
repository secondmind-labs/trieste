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
    ActivityCondition,
    BooleanSearchSpace,
    Box,
    CategoricalSearchSpace,
    DiscreteSearchSpace,
    HierarchicalSearchSpace,
    HierarchyNode,
    SearchSpace,
    hierarchy_node_from_tags,
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


def test_helper_translates_activity_condition_tags_to_global_columns() -> None:
    subspaces = _worked_example_subspaces()
    node = hierarchy_node_from_tags(
        "branch_A",
        subspace_tags=["x2"],
        activity_condition_tags={"y1": 1},
        subspaces=subspaces,
        indicator_tags=["y1"],
    )
    # y1 is at flat-vector column 1 (x1=0, y1=1, ...), and that column is the key.
    assert node.activity_condition.requirements == {1: 1}


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
    # Value 2 must survive as int (not bool-coerced to True). y1 is at column 1.
    req = node.activity_condition.requirements
    assert req[1] == 2
    assert not isinstance(req[1], bool)


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


def test_helper_infers_indicator_when_indicator_tags_omitted() -> None:
    # Omitting indicator_tags: the activity_condition_tags key self-identifies as an indicator
    # and is resolved to its global column (y1 -> column 1) from subspaces alone.
    subspaces = _worked_example_subspaces()
    node = hierarchy_node_from_tags(
        "branch_A",
        subspace_tags=["x2"],
        activity_condition_tags={"y1": 1},
        subspaces=subspaces,
    )
    assert node.activity_condition.requirements == {1: 1}


def test_helper_explicit_indicator_tags_still_catches_unknown_key() -> None:
    # When indicator_tags is supplied it remains a cross-check against typos.
    subspaces = _worked_example_subspaces()
    with pytest.raises(ValueError, match="is not in"):
        hierarchy_node_from_tags(
            "branch_A",
            subspace_tags=["x2"],
            activity_condition_tags={"x4": 1},  # x4 is not declared an indicator
            subspaces=subspaces,
            indicator_tags=["y1"],
        )


def test_helper_rejects_activity_condition_key_not_a_subspace() -> None:
    subspaces = _worked_example_subspaces()
    with pytest.raises(ValueError, match="not a key of"):
        hierarchy_node_from_tags(
            "branch_A",
            subspace_tags=["x2"],
            activity_condition_tags={"nope": 1},
            subspaces=subspaces,
        )


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
        hierarchy_node_from_tags(
            "shared", subspace_tags=["x1"], subspaces=subspaces, indicator_tags=["y1", "y2"]
        ),
        hierarchy_node_from_tags(
            "branch",
            subspace_tags=["x2"],
            activity_condition_tags={"y1": 1, "y2": 0},
            subspaces=subspaces,
            indicator_tags=["y1", "y2"],
        ),
    ]
    space = HierarchicalSearchSpace(subspaces, hierarchy, indicator_tags=["y1", "y2"])
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
        hierarchy_node_from_tags("shared", subspace_tags=["x1", "b"], subspaces=subspaces),
        hierarchy_node_from_tags(
            "branch", subspace_tags=["x2"], activity_condition_tags={"y1": 1}, subspaces=subspaces
        ),
    ]
    space = HierarchicalSearchSpace(subspaces, hierarchy)  # indicator_tags inferred
    assert space.indicator_tags == ("y1",)
    assert "b" in space.non_indicator_tags


def test_hss_explicit_declaring_a_feature_boolean_as_indicator_errors() -> None:
    # Same space as the inference test above: omitting indicator_tags treats ``b`` as a feature.
    # The explicit override is the safety net -- declaring the feature Boolean ``b`` an indicator
    # is rejected (its column is already a feature_dim, so it cannot also be an indicator).
    subspaces = {
        "x1": Box([0.0], [1.0]),
        "b": BooleanSearchSpace(),
        "y1": BooleanSearchSpace(),
        "x2": Box([0.0], [1.0]),
    }
    hierarchy = [
        hierarchy_node_from_tags("shared", subspace_tags=["x1", "b"], subspaces=subspaces),
        hierarchy_node_from_tags(
            "branch", subspace_tags=["x2"], activity_condition_tags={"y1": 1}, subspaces=subspaces
        ),
    ]
    with pytest.raises(ValueError, match="indicator column"):
        HierarchicalSearchSpace(subspaces, hierarchy, indicator_tags=["y1", "b"])


def test_hss_inferred_box_indicator_rejected_by_type_check() -> None:
    # If a node gates on a Box column, inference treats it as an indicator and the type check
    # rejects it (indicators must be Boolean / 1-D categorical).
    subspaces = {"x1": Box([0.0], [1.0]), "x2": Box([0.0], [1.0])}
    hierarchy = [
        hierarchy_node_from_tags(
            "n", subspace_tags=["x1"], activity_condition_tags={"x2": 1}, subspaces=subspaces
        )
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
    hierarchy = [hierarchy_node_from_tags("only", subspace_tags=["x1"], subspaces=subspaces)]
    space = HierarchicalSearchSpace(subspaces, hierarchy, indicator_tags=[])
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
        hierarchy_node_from_tags(
            "shared",
            subspace_tags=["x1"],
            subspaces=subspaces,
            indicator_tags=["y1", "y2"],
        ),
        hierarchy_node_from_tags(
            "a",
            subspace_tags=["x2"],
            activity_condition_tags={"y1": 1},
            subspaces=subspaces,
            indicator_tags=["y1", "y2"],
        ),
        hierarchy_node_from_tags(
            "b",
            subspace_tags=["x3"],
            activity_condition_tags={"y2": 1},
            subspaces=subspaces,
            indicator_tags=["y1", "y2"],
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


def test_hss_raises_if_indicator_tag_not_a_key_of_subspaces() -> None:
    subspaces = {"x1": Box([0.0], [1.0]), "y1": BooleanSearchSpace()}
    hierarchy = [
        hierarchy_node_from_tags(
            "n",
            subspace_tags=["x1"],
            activity_condition_tags={"y1": 1},
            subspaces=subspaces,
            indicator_tags=["y1"],
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


@pytest.mark.parametrize(
    "feature_dims, feature_bounds, requirements, match",
    [
        # feature_dims points at y1, which is an indicator column.
        pytest.param([1], [[0.0, 1.0]], {1: 1}, "indicator column", id="feature_dim_is_indicator"),
        pytest.param([42], [[0.0, 1.0]], {1: 1}, "out of range", id="feature_dim_out_of_range"),
        # x1 lives in [0, 1] but the node claims [-99, 99].
        pytest.param([0], [[-99.0, 99.0]], {1: 1}, "feature_bounds", id="bounds_disagree"),
        # The only indicator (y1) is at column 1, but the node references column 7.
        pytest.param([0], [[0.0, 1.0]], {7: 1}, "not an indicator column", id="key_not_indicator"),
    ],
)
def test_hss_construction_rejects_invalid_nodes(
    feature_dims: list[int],
    feature_bounds: list[list[float]],
    requirements: dict[int, int],
    match: str,
) -> None:
    subspaces = {"x1": Box([0.0], [1.0]), "y1": BooleanSearchSpace()}
    bad_node = gpflow.kernels.HierarchyNode(
        "n",
        feature_dims=feature_dims,
        feature_bounds=tf.constant(feature_bounds, dtype=tf.float64),
        activity_condition=ActivityCondition(requirements),
    )
    with pytest.raises(ValueError, match=match):
        HierarchicalSearchSpace(subspaces, [bad_node], indicator_tags=["y1"])


def test_hss_raises_if_required_value_not_in_permitted_set() -> None:
    subspaces = {"x1": Box([0.0], [1.0]), "y1": CategoricalSearchSpace(3)}
    bad_node = gpflow.kernels.HierarchyNode(
        "n",
        feature_dims=[0],
        feature_bounds=tf.constant([[0.0, 1.0]], dtype=tf.float64),
        activity_condition=ActivityCondition({1: 5}),  # y1 at column 1; K=3, so 5 invalid
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
            "n",
            subspace_tags=["x1"],
            activity_condition_tags={"y1": 1},
            subspaces=subspaces,
            indicator_tags=["y1"],
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
            "n",
            subspace_tags=["x1"],
            activity_condition_tags={"y1": 1},
            subspaces=subspaces,
            indicator_tags=["y1", "y2"],
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
            "shared",
            subspace_tags=["x1"],
            subspaces=subspaces,
            indicator_tags=["y1"],
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
    return HierarchicalSearchSpace(subspaces, hierarchy, indicator_tags=["y1"])


def test_categorical_hss_enumerate_tasks_three_configs() -> None:
    tasks = _make_categorical_hss().enumerate_tasks()
    assert tasks == [{"y1": 0}, {"y1": 1}, {"y1": 2}]


def test_categorical_hss_active_subspaces() -> None:
    space = _make_categorical_hss()
    assert space.active_subspace_tags({"y1": 0}) == ["x1"]
    assert set(space.active_subspace_tags({"y1": 1})) == {"x1", "x2"}
    assert set(space.active_subspace_tags({"y1": 2})) == {"x1", "x3"}


# ===== additional validation =====


def test_hss_raises_if_activity_condition_key_negative() -> None:
    subspaces = {"x1": Box([0.0], [1.0]), "y1": BooleanSearchSpace()}
    # ActivityCondition validates non-negative keys on construction, so bypass it
    # to exercise the HierarchicalSearchSpace defence: a negative column is not an
    # indicator column and must be rejected.
    condition = ActivityCondition({1: 1})
    object.__setattr__(condition, "requirements", {-1: 1})
    bad_node = gpflow.kernels.HierarchyNode(
        "n",
        feature_dims=[0],
        feature_bounds=tf.constant([[0.0, 1.0]], dtype=tf.float64),
        activity_condition=condition,
    )
    with pytest.raises(ValueError, match="not an indicator column"):
        HierarchicalSearchSpace(subspaces, [bad_node], indicator_tags=["y1"])


def test_hss_raises_if_non_indicator_subspace_has_no_bounds() -> None:
    # c1 is a non-indicator categorical subspace: it has no numerical bounds, so
    # it cannot be encoded as a (lower, upper) feature_bounds row.
    subspaces = {
        "x1": Box([0.0], [1.0]),
        "c1": CategoricalSearchSpace(3),
        "y1": BooleanSearchSpace(),
    }
    node = gpflow.kernels.HierarchyNode(
        "n",
        feature_dims=[0],
        feature_bounds=tf.constant([[0.0, 1.0]], dtype=tf.float64),
        activity_condition=ActivityCondition({0: 1}),
    )
    with pytest.raises(ValueError, match="no numerical bounds"):
        HierarchicalSearchSpace(subspaces, [node], indicator_tags=["y1"])


def test_helper_raises_on_duplicate_subspace_tags() -> None:
    subspaces = {"x1": Box([0.0], [1.0]), "y1": BooleanSearchSpace()}
    with pytest.raises(ValueError, match="duplicate tags"):
        hierarchy_node_from_tags(
            "n",
            subspace_tags=["x1", "x1"],
            activity_condition_tags={"y1": 1},
            subspaces=subspaces,
            indicator_tags=["y1"],
        )


def test_helper_rejects_empty_subspace_tags() -> None:
    subspaces = {"x1": Box([0.0], [1.0]), "y1": BooleanSearchSpace()}
    with pytest.raises(ValueError, match="must be non-empty"):
        hierarchy_node_from_tags("n", subspace_tags=[], subspaces=subspaces)


def test_helper_rejects_subspace_tag_not_a_subspace() -> None:
    subspaces = {"x1": Box([0.0], [1.0]), "y1": BooleanSearchSpace()}
    with pytest.raises(ValueError, match="not a key of"):
        hierarchy_node_from_tags("n", subspace_tags=["missing"], subspaces=subspaces)


def test_helper_rejects_multidimensional_indicator() -> None:
    # An activity-condition key resolving to more than one column is not a valid indicator.
    subspaces = {"x1": Box([0.0], [1.0]), "big": Box([0.0, 0.0], [1.0, 1.0])}
    with pytest.raises(ValueError, match="must be 1-dimensional"):
        hierarchy_node_from_tags(
            "n", subspace_tags=["x1"], activity_condition_tags={"big": 1}, subspaces=subspaces
        )


def test_helper_raises_on_overlapping_subspace_and_indicator_tags() -> None:
    subspaces = {"x1": Box([0.0], [1.0]), "y1": BooleanSearchSpace()}
    with pytest.raises(ValueError, match="must be disjoint"):
        hierarchy_node_from_tags(
            "n",
            subspace_tags=["y1"],  # y1 is also declared an indicator below
            subspaces=subspaces,
            indicator_tags=["y1"],
        )


def test_helper_rejects_non_indicator_subspace_without_bounds() -> None:
    # A categorical (no numerical bounds) cannot be a non-indicator feature.
    subspaces = {
        "x1": Box([0.0], [1.0]),
        "c": CategoricalSearchSpace(3),
        "y1": BooleanSearchSpace(),
    }
    with pytest.raises(ValueError, match="without numerical"):
        hierarchy_node_from_tags("n", subspace_tags=["c"], subspaces=subspaces)


def test_helper_accepts_discrete_non_indicator_subspace() -> None:
    # A discrete (bounded) subspace is allowed as a non-indicator feature.
    subspaces = {
        "x1": Box([0.0], [1.0]),
        "d": DiscreteSearchSpace(tf.constant([[0.0], [1.0], [2.0]], dtype=tf.float64)),
        "y1": BooleanSearchSpace(),
    }
    node = hierarchy_node_from_tags("shared", subspace_tags=["x1", "d"], subspaces=subspaces)
    # x1 -> col 0, d -> col 1; bounds reflect each subspace.
    assert list(node.feature_dims) == [0, 1]
    bounds = tf.convert_to_tensor(node.feature_bounds, dtype=tf.float64).numpy()
    npt.assert_array_almost_equal(bounds, [[0.0, 1.0], [0.0, 2.0]])


def test_hss_raises_on_duplicate_indicator_tags() -> None:
    subspaces = {"x1": Box([0.0], [1.0]), "y1": BooleanSearchSpace()}
    node = hierarchy_node_from_tags(
        "n",
        subspace_tags=["x1"],
        activity_condition_tags={"y1": 1},
        subspaces=subspaces,
        indicator_tags=["y1"],
    )
    with pytest.raises(ValueError, match="duplicate tags"):
        HierarchicalSearchSpace(subspaces, [node], indicator_tags=["y1", "y1"])


# ===== __eq__ / __repr__ =====


def test_hss_eq_true_for_identical_spaces() -> None:
    assert _make_worked_example_hss() == _make_worked_example_hss()


def test_hss_eq_false_when_hierarchy_differs() -> None:
    # Same subspaces, but a hierarchy whose node names differ.
    subspaces = _worked_example_subspaces()
    renamed_hierarchy = [
        hierarchy_node_from_tags(
            "shared_renamed",
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
    other = HierarchicalSearchSpace(subspaces, renamed_hierarchy, indicator_tags=["y1"])
    assert _make_worked_example_hss() != other


def test_hss_eq_false_for_non_hierarchical_object() -> None:
    assert _make_worked_example_hss() != 5


def test_hss_repr_includes_hierarchy_and_indicator_tags() -> None:
    text = repr(_make_worked_example_hss())
    assert "hierarchy" in text
    assert "indicator_tags" in text
    assert "y1" in text


# ===== gpflow kernel integration =====


def test_hss_hierarchy_requirements_use_global_columns() -> None:
    space = _make_worked_example_hss()
    # Columns: [x1, y1, x2, x4, x3]; the single indicator y1 is at flat column 1,
    # and requirements are keyed by that global column (gpflow's convention).
    assert space.indicator_dims == [1]
    by_name = {n.name: n for n in space.hierarchy}
    assert dict(by_name["shared"].activity_condition.requirements) == {}
    assert dict(by_name["branch_A"].activity_condition.requirements) == {1: 1}
    assert dict(by_name["branch_B"].activity_condition.requirements) == {1: 0}


def test_hss_hierarchy_is_directly_consumable_by_arc_hierarchical() -> None:
    # No adapter needed: space.hierarchy already uses gpflow's column convention,
    # so feature columns and indicator columns tile active_dims contiguously.
    space = _make_worked_example_hss()
    active_dims = list(range(int(space.dimension)))
    kernel = gpflow.kernels.ArcHierarchical(
        list(space.hierarchy), active_dims=active_dims
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
