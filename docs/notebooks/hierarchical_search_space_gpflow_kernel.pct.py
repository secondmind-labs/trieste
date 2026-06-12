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
# # A GPflow hierarchical kernel over a `HierarchicalSearchSpace`
#
# The previous notebook introduced `HierarchicalSearchSpace`, the conditional /
# disjunctive search space that gates non-indicator subspaces by indicator
# values. To do Bayesian optimisation over such a space we need a covariance
# function that respects the activation structure: a point whose conditional
# feature is *inactive* should not be confused with one whose feature is
# *active and equal in value* — the two live in fundamentally different
# regions of the space.
#
# GPflow ships such kernels: `gpflow.kernels.ArcHierarchical` (the **Arc**
# kernel of Swersky et al. 2014) and `gpflow.kernels.WedgeHierarchical` (the
# **Wedge** kernel of Horn et al. 2019). This notebook shows how to feed a
# Trieste `HierarchicalSearchSpace` straight into `ArcHierarchical` and fit a
# GP — i.e. the integration glue between Trieste's search-space layer and
# GPflow's kernel layer. The only real work is a small coordinate-system
# adapter, explained below.

# %%
import gpflow
import numpy as np
import tensorflow as tf
from gpflow.kernels import ArcHierarchical, WedgeHierarchical

from trieste.space import (
    BooleanSearchSpace,
    Box,
    HierarchicalSearchSpace,
    hierarchy_node_from_tags,
)

np.random.seed(1793)
tf.random.set_seed(1793)

# %% [markdown]
# ## Three axioms for a conditional distance
#
# Let $\mathbf{x}^{nc}$ be the always-active (unconditional) coordinates and
# $\mathbf{x}^{c}$ the indicator-gated (conditional) ones. A useful per-dimension
# distance $d_i(\mathbf{x}, \mathbf{x}')$ on a *conditional* dimension $i$
# should satisfy:
#
# 1. **both inactive:** $d_i = 0$ — the value is meaningless on either side.
# 2. **both active:** $d_i$ is a function of the difference in feature value.
# 3. **incomparable** (one active, one inactive): $d_i$ is positive and places
#    the two points in distinct regions of the embedded space.
#
# Both `ArcHierarchical` and `WedgeHierarchical` enforce these axioms by
# **embedding each conditional dimension into $\mathbb{R}^2$**: inactive points
# map to the origin, active points to a value-dependent point off the origin. A
# stationary base kernel (Matérn-5/2 by default) then evaluates the covariance
# in the joint embedded space. We do not have to implement any of this — we
# only have to describe the activation structure to the kernel.

# %% [markdown]
# ## Build the search space
#
# Canonical four-variable disjunction: $x_1$ unconditional, $y_1$ a Boolean
# indicator, $x_2$ active when $y_1 = 1$, $x_3$ active when $y_1 = 0$. This is
# the same space as the previous notebook.

# %%
subspaces = {
    "x1": Box([0.0], [1.0]),  # unconditional
    "y1": BooleanSearchSpace(),  # Boolean indicator
    "x2": Box([0.0], [5.0]),  # active when y1 = 1
    "x3": Box([-1.0], [1.0]),  # active when y1 = 0
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
        activity_condition_tags={"y1": 0},
        subspaces=subspaces,
        indicator_tags=["y1"],
    ),
]
space = HierarchicalSearchSpace(subspaces, hierarchy, indicator_tags=["y1"])
print("dimension:     ", int(space.dimension))
print("indicator_tags:", space.indicator_tags)
print("indicator_dims:", space.indicator_dims)

# %% [markdown]
# ## From a `HierarchicalSearchSpace` to a GPflow hierarchy
#
# `ArcHierarchical` takes a sequence of `gpflow.kernels.HierarchyNode` plus an
# `active_dims` argument:
#
# ```python
# ArcHierarchical(hierarchy, base_kernel=None, *, active_dims, name=None)
# ```
#
# Each node carries `feature_dims` (the real-valued columns it owns),
# `feature_bounds` (for normalisation), and an `ActivityCondition` whose
# `requirements` map gates those columns. The kernel **infers** the indicator
# columns as the union of all `requirements` keys, slices its input down to
# `active_dims`, and validates that the feature columns and the inferred
# indicator columns together tile the sliced vector contiguously.
#
# `HierarchicalSearchSpace.hierarchy` exposes exactly these `HierarchyNode`
# objects: Trieste keys both `feature_dims` and the `requirements` map by global
# flat-vector column (the same convention the kernel uses), so the nodes can be
# passed straight to `ArcHierarchical` — no adapter needed. Since every column of
# an HSS flat vector is either a feature or an indicator, we slice on **all**
# columns — `active_dims = range(dimension)` — so the sliced and global
# coordinate systems coincide.

# %%
active_dims = list(range(int(space.dimension)))
for node in space.hierarchy:
    print(
        f"{node.name:9s} feature_dims={list(node.feature_dims)} "
        f"requirements={dict(node.activity_condition.requirements)}"
    )
print("active_dims:", active_dims)

# %% [markdown]
# ## Construct the Arc kernel
#
# With the hierarchy and `active_dims` in hand, the kernel is a one-liner. (The
# hierarchical kernels are flagged `@experimental` in GPflow 2.11, so
# constructing one emits a `UserWarning` — that is expected.)

# %%
arc = ArcHierarchical(list(space.hierarchy), active_dims=active_dims)
print(type(arc).__name__, "built over active_dims", active_dims)

# %% [markdown]
# ## Inspect the covariance on a hand-crafted batch
#
# The flat-vector layout is `[x1, y1, x2, x3]`. The two rows below share
# $x_1 = 0.5$ and the *same stored* $x_2 = 2.5$, but differ in the indicator:
# the first has $y_1 = 1$ (so $x_2$ is active, $x_3$ inactive) and the second
# has $y_1 = 0$ (so $x_3$ is active, $x_2$ inactive). A conditional kernel must
# place them in different regions, so their cross-covariance is well below the
# unit diagonal.

# %%
X_demo = tf.constant(
    [
        [0.5, 1.0, 2.5, 0.0],  # y1 = 1: x1 + x2 active
        [0.5, 0.0, 2.5, 0.0],  # y1 = 0: x1 + x3 active
    ],
    dtype=tf.float64,
)
print("K(X_demo):")
print(arc.K(X_demo).numpy())

# %% [markdown]
# ## Worked example: fit a GP on a synthetic disjunctive function
#
# Synthetic objective whose two branches carry different signal:
#
# $$f(x_1, y_1, x_2, x_3) =
#     \sin(2\pi x_1)
#   \;+\; \mathbf{1}[y_1 = 1] \cdot \tfrac{1}{2} \cos(\pi x_2 / 5)
#   \;+\; \mathbf{1}[y_1 = 0] \cdot \tfrac{1}{2} x_3.$$
#
# The conditional kernel must "switch off" the inactive branch's contribution
# to similarity for that to be learnable from a finite sample.


# %%
def objective(X):
    X = np.asarray(X)
    x1, y1, x2, x3 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    return (
        np.sin(2.0 * np.pi * x1)
        + (y1 > 0.5).astype(float) * 0.5 * np.cos(np.pi * x2 / 5.0)
        + (y1 < 0.5).astype(float) * 0.5 * x3
    ).reshape(-1, 1)


X_train = space.sample(40).numpy()
Y_train = objective(X_train) + 0.05 * np.random.randn(40, 1)

# Wrap in a Constant() factor so the GP can learn an overall variance.
kernel = gpflow.kernels.Constant() * ArcHierarchical(
    list(space.hierarchy), active_dims=active_dims
)
gpr = gpflow.models.GPR(
    data=(X_train, Y_train), kernel=kernel, noise_variance=0.05
)

print(f"LML before fit: {gpr.log_marginal_likelihood().numpy():+.3f}")
gpflow.optimizers.Scipy().minimize(
    gpr.training_loss, gpr.trainable_variables, options={"maxiter": 100}
)
print(f"LML after fit:  {gpr.log_marginal_likelihood().numpy():+.3f}")

# %% [markdown]
# ## Predict on held-out points
#
# We evaluate on a small held-out batch covering both branches. The kernel
# places the two test rows on different parts of the embedded space, so their
# predicted means follow the corresponding branch's signal.

# %%
X_test = np.array(
    [
        [0.3, 1.0, 2.0, 0.0],  # y1 = 1
        [0.3, 0.0, 0.0, 0.5],  # y1 = 0
        [0.7, 1.0, 4.0, 0.0],  # y1 = 1
        [0.7, 0.0, 0.0, -0.4],  # y1 = 0
    ]
)
mean, var = gpr.predict_f(X_test)
print("test predictions vs ground truth:")
truth = objective(X_test).ravel()
for x, m, v, t in zip(X_test, mean.numpy().ravel(), var.numpy().ravel(), truth):
    print(
        f"  x = {x.tolist()}  ->  mean = {m:+.3f}  var = {v:.3f}  truth = {t:+.3f}"
    )

# %% [markdown]
# ## Swapping in the Wedge kernel
#
# `WedgeHierarchical` has the identical constructor, so switching embeddings is
# a one-line change. Its "incomparable" distance scales with the active value
# $v_c$ rather than being constant in it — often closer to what a practitioner
# expects near disjunction boundaries.

# %%
wedge = WedgeHierarchical(list(space.hierarchy), active_dims=active_dims)
print("Wedge kernel matrix on demo points:")
print(wedge.K(X_demo).numpy())

# %% [markdown]
# ## What this shows
#
# A conditional GP over a disjunctive space needs only:
#
# * `BooleanSearchSpace`, `Box`, `HierarchicalSearchSpace`,
#   `hierarchy_node_from_tags` from `trieste.space` to describe the structure;
# * `list(space.hierarchy)` passed straight to the kernel — Trieste keys the
#   hierarchy in GPflow's column convention, so no adapter is needed;
# * `gpflow.kernels.ArcHierarchical` (or `WedgeHierarchical`) for the covariance.
#
# No bespoke kernel code is required — the hierarchy carried by
# `HierarchicalSearchSpace` is exactly the information GPflow's hierarchical
# kernels consume.
