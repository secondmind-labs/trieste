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

import numpy.testing as npt
import pytest
import tensorflow as tf

from trieste.space import (
    BooleanSearchSpace,
    Box,
    CategoricalSearchSpace,
    DiscreteSearchSpace,
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
