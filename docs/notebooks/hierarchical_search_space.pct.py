# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: .venv_310
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Hierarchical search spaces
#
# This notebook demonstrates Trieste's `HierarchicalSearchSpace`, a search space that
# represents the conditional activation structure of a Generalized Disjunctive Program (GDP).
# Variables in such a space are gated by *indicator variables* (binary or categorical)
# that decide which other dimensions are meaningful.
#
# The core primitives introduced in PR 1 of the hierarchical-inputs effort are:
#
# - `BooleanSearchSpace`: a one-dimensional discrete space restricted to ``{0, 1}``.
# - `CategoricalSearchSpace`: a one-dimensional discrete space over ``K`` named or
#   indexed categories.
# - `HierarchyNode`: a frozen dataclass describing one disjunction (which subspaces are
#   active when which indicators take which values).
# - `HierarchicalSearchSpace`: a `CollectionSearchSpace` that wires the indicators and
#   nodes together and exposes a tag-based query API for downstream consumers.
#
# We walk through the API on two small examples: a Boolean indicator (the canonical
# four-variable disjunction) and a 3-ary categorical indicator. The notebook stops at
# the search-space layer; an end-to-end Bayesian-optimisation tutorial that combines
# `HierarchicalSearchSpace` with the Conditional kernel and the GA acquisition
# optimiser will land alongside PR 4 (kernels) and PR 6 (benchmarks).

# %%
import numpy as np
import tensorflow as tf

from trieste.space import (
    Box,
    BooleanSearchSpace,
    CategoricalSearchSpace,
    HierarchyNode,
    HierarchicalSearchSpace,
)

np.random.seed(1793)
tf.random.set_seed(1793)

# %% [markdown]
# ## Boolean-indicator example
#
# Consider a four-variable space with one Boolean indicator. Variable $x_1$ is
# continuous and always active; $y_1$ is the indicator; $x_2$ is continuous and active
# only when $y_1 = 1$; $x_3$ is continuous and active only when $y_1 = 0$.
#
# We list the subspaces, give them tags, then describe the hierarchy as three
# `HierarchyNode`s — one unconditional ("shared") and two gated by `y1`.

# %%
spaces = [
    Box([0.0], [1.0]),  # x1: unconditional
    BooleanSearchSpace(),  # y1: Boolean indicator
    Box([0.0], [5.0]),  # x2: active when y1 = 1
    Box([-1.0], [1.0]),  # x3: active when y1 = 0
]
tags = ["x1", "y1", "x2", "x3"]
hierarchy = [
    HierarchyNode("shared", subspace_tags=["x1"], indicator_conditions={}),
    HierarchyNode("branch_A", subspace_tags=["x2"], indicator_conditions={"y1": True}),
    HierarchyNode("branch_B", subspace_tags=["x3"], indicator_conditions={"y1": False}),
]
space = HierarchicalSearchSpace(spaces, tags, hierarchy, indicator_tags=["y1"])

print("dimension:", int(space.dimension))
print("indicator_tags:", space.indicator_tags)
print("non_indicator_tags:", space.non_indicator_tags)
print("indicator_value_sets:", space.indicator_value_sets)

# %% [markdown]
# ### Sampling and the flat-vector layout
#
# Points are represented as flat vectors `[x1, y1, x2, x3]` concatenated in tag order,
# the same convention as `TaggedProductSearchSpace`. For a sample with $y_1 = 1$ the
# `x3` column is *inactive*: its value is still present in the vector but has no
# semantic meaning. Downstream kernels (Arc, Wedge, Conditional) handle inactive
# columns according to their respective axioms.

# %%
samples = space.sample(8)
print("samples shape:", samples.shape)
print(samples.numpy())

# %% [markdown]
# `get_subspace_component` slices the columns belonging to a given tag. It works on
# any batch shape because indexing is on the trailing axis.

# %%
print("y1 column:", space.get_subspace_component("y1", samples).numpy().ravel())
print("x2 column:", space.get_subspace_component("x2", samples).numpy().ravel())

# %% [markdown]
# ### Hierarchy queries
#
# Several methods expose the hierarchy programmatically. `enumerate_tasks` returns
# every indicator configuration; `active_subspace_tags` returns the non-indicator
# tags active for one configuration; `is_active` answers the per-tag question; and
# `node_for_subspace` returns the nodes that contain a given tag.

# %%
print("enumerate_tasks:", space.enumerate_tasks())
print("active for y1=1:", space.active_subspace_tags({"y1": 1}))
print("active for y1=0:", space.active_subspace_tags({"y1": 0}))
print("x2 is_active when y1=1:", space.is_active("x2", {"y1": 1}))
print("x2 is_active when y1=0:", space.is_active("x2", {"y1": 0}))
print("nodes containing 'x1':", [n.name for n in space.node_for_subspace("x1")])
print("nodes containing 'x2':", [n.name for n in space.node_for_subspace("x2")])

# %% [markdown]
# ## Categorical-indicator example
#
# When a disjunction has more than two branches, a single $K$-ary categorical
# indicator is the natural representation. Suppose $y_1$ now selects among three unit
# types $\{0\colon\text{shared-only},\;1\colon\text{branch A},\;2\colon\text{branch B}\}$.

# %%
spaces_c = [
    Box([0.0], [1.0]),  # x1: unconditional
    CategoricalSearchSpace(3),  # y1: 3-ary indicator
    Box([0.0], [5.0]),  # x2: active when y1 = 1
    Box([-1.0], [1.0]),  # x3: active when y1 = 2
]
hierarchy_c = [
    HierarchyNode("shared", subspace_tags=["x1"], indicator_conditions={}),
    HierarchyNode("branch_A", subspace_tags=["x2"], indicator_conditions={"y1": 1}),
    HierarchyNode("branch_B", subspace_tags=["x3"], indicator_conditions={"y1": 2}),
]
space_c = HierarchicalSearchSpace(spaces_c, tags, hierarchy_c, indicator_tags=["y1"])

print("indicator_value_sets:", space_c.indicator_value_sets)
print("enumerate_tasks:", space_c.enumerate_tasks())
print("active for y1=0:", space_c.active_subspace_tags({"y1": 0}))
print("active for y1=1:", space_c.active_subspace_tags({"y1": 1}))
print("active for y1=2:", space_c.active_subspace_tags({"y1": 2}))

# %% [markdown]
# ## Mixing indicator kinds
#
# Boolean and categorical indicators can be combined in a single
# `HierarchicalSearchSpace`. `enumerate_tasks` then returns the Cartesian product of
# every indicator's permitted set: $|\{0,1\}| \times |\{0,1,2\}| = 6$ tasks below.

# %%
spaces_mix = [
    Box([0.0], [1.0]),
    BooleanSearchSpace(),
    CategoricalSearchSpace(3),
    Box([0.0], [5.0]),
]
tags_mix = ["x1", "y1", "y2", "x2"]
hierarchy_mix = [
    HierarchyNode("shared", subspace_tags=["x1"], indicator_conditions={}),
    HierarchyNode("branch", subspace_tags=["x2"], indicator_conditions={"y1": True, "y2": 2}),
]
space_mix = HierarchicalSearchSpace(
    spaces_mix, tags_mix, hierarchy_mix, indicator_tags=["y1", "y2"]
)
tasks = space_mix.enumerate_tasks()
print(f"number of tasks: {len(tasks)}")
for t in tasks:
    print(" ", t)

# %% [markdown]
# ## Validation rules
#
# Construction fails fast with a `ValueError` when the inputs are inconsistent. A few
# representative rejections:
#
# - an indicator tag that does not point to a `BooleanSearchSpace` or a
#   one-dimensional `CategoricalSearchSpace`;
# - a `HierarchyNode.indicator_conditions` value that is outside the indicator's
#   permitted set (e.g. `5` for a 3-ary categorical, or any non-Boolean value for a
#   `BooleanSearchSpace`);
# - an indicator that gates nothing (does not appear as a key in any node).


# %%
def _expect_value_error(fn):
    try:
        fn()
    except ValueError as e:
        print("rejected:", str(e).split("\n")[0])
    else:
        raise AssertionError("expected ValueError but none was raised")


_expect_value_error(
    lambda: HierarchicalSearchSpace(
        spaces=[Box([0.0], [1.0]), CategoricalSearchSpace(3)],
        tags=["x1", "y1"],
        hierarchy=[HierarchyNode("n", subspace_tags=["x1"], indicator_conditions={"y1": 5})],
        indicator_tags=["y1"],
    )
)

_expect_value_error(
    lambda: HierarchicalSearchSpace(
        spaces=[Box([0.0], [1.0]), CategoricalSearchSpace([3, 2])],
        tags=["x1", "y1"],
        hierarchy=[HierarchyNode("n", subspace_tags=["x1"], indicator_conditions={"y1": 1})],
        indicator_tags=["y1"],
    )
)

# %% [markdown]
# ## What's next
#
# This notebook covers the PR 1 surface only. PR 1b adds:
#
# - `ConditionalConstraint` for indicator-gated disjunctive constraints
#   $h_{ik}(\mathbf{x}) \le 0$;
# - `LogicalProposition` for consistency conditions $\Omega(\mathbf{Y})$ on the
#   indicators alone;
# - constraint integration on `HierarchicalSearchSpace`
#   (`global_constraints`, `conditional_constraints`, `logical_propositions`,
#   `constraints_residuals`, `is_feasible`).
#
# Once PR 1b is merged this notebook will be extended with a constrained example
# (a global linear coupling, a conditional inequality gated by an indicator, and a
# logical proposition on the indicators). A richer end-to-end Bayesian-optimisation
# tutorial that combines `HierarchicalSearchSpace` with the Conditional kernel and
# the GA acquisition optimiser will land alongside PR 4 (kernels) and PR 6
# (benchmarks).
