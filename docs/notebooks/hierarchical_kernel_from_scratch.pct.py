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
# # Arc and Wedge kernels from scratch over a `HierarchicalSearchSpace`
#
# The previous notebook introduced `HierarchicalSearchSpace`, the conditional /
# disjunctive search space that gates non-indicator subspaces by indicator
# values. To do Bayesian optimisation over such a space we need a covariance
# function that respects the activation structure: a point whose conditional
# feature is *inactive* should not be confused with one whose feature is
# *active and equal in value* — the two live in fundamentally different
# regions of the space.
#
# This notebook implements two such kernels — the **Arc** kernel (Swersky et al.
# 2014) and the **Wedge** kernel (Horn et al. 2019) — *from scratch*, using
# only:
#
# * the search-space primitives in `trieste.space`,
# * GPflow's stationary kernels as the base covariance in the embedded space,
# * TensorFlow / TensorFlow Probability for differentiable plumbing.
#
# Nothing from `trieste.models` is used. The point is to show that the
# `HierarchicalSearchSpace` API alone is enough to wire a conditional GP, and
# to give a self-contained reading of the kernel before its added to relevant
# tooling.

# %%
import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.utilities import positive, to_default_float

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
# A clean way to enforce these axioms is to **embed each conditional dimension
# into $\mathbb{R}^2$**: inactive points map to the origin, active points to a
# value-dependent point off the origin. A stationary base kernel then
# evaluates the covariance in the joint embedded space.

# %% [markdown]
# ## Build the search space
#
# Canonical four-variable disjunction: $x_1$ unconditional, $y_1$ a Boolean
# indicator, $x_2$ active when $y_1 = 1$, $x_3$ active when $y_1 = 0$.

# %%
subspaces = {
    "x1": Box([0.0], [1.0]),  # unconditional
    "y1": BooleanSearchSpace(),  # Boolean indicator
    "x2": Box([0.0], [5.0]),  # active when y1 = 1
    "x3": Box([-1.0], [1.0]),  # active when y1 = 0
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
        "branch_B", subspace_tags=["x3"], activity_condition_tags={"y1": 0},
        subspaces=subspaces, indicator_tags=["y1"],
    ),
]
space = HierarchicalSearchSpace(subspaces, hierarchy, indicator_tags=["y1"])
print("dimension:", int(space.dimension))
print("indicator_tags:", space.indicator_tags)
print("non_indicator_tags:", space.non_indicator_tags)

# %% [markdown]
# ## Translate the search space into kernel primitives
#
# The kernel does not need to know about tags or `HierarchyNode`s at evaluation
# time — only:
#
# * **`feature_dims`** — column indices of real-valued (non-indicator) features
#   in the flat input vector;
# * **`feature_bounds`** — a `[D_f, 2]` tensor of `(lower, upper)` per feature
#   (used for normalisation);
# * **`indicator_dims`** — column indices of the indicator values;
# * **`activity_conditions`** — for each non-indicator feature column, a dict
#   `{local_indicator_index: required_value}` representing the AND-conjunction
#   of indicator requirements (an empty dict means the column is unconditional).
#
# We walk `space.subspace_tags` in order to discover column positions, then
# read the activity rules off `space.node_for_subspace(tag)`, which returns
# `gpflow.kernels.HierarchyNode` objects. The integer-keyed
# `node.activity_condition.requirements` map is the source of truth for the
# AND-conjunction; no tag→index translation is needed.


# %%
def primitives_from_space(space):
    indicator_set = set(space.indicator_tags)

    indicator_dims, feature_dims = [], []
    lowers, uppers, activity_conditions = [], [], []
    col = 0

    for tag in space.subspace_tags:
        sub = space.get_subspace(tag)
        sub_dim = int(sub.dimension)
        if tag in indicator_set:
            assert sub_dim == 1, "indicators must be one-dimensional"
            indicator_dims.append(col)
        else:
            feature_dims.extend(range(col, col + sub_dim))
            lowers.append(tf.cast(sub.lower, gpflow.default_float()))
            uppers.append(tf.cast(sub.upper, gpflow.default_float()))
            nodes = space.node_for_subspace(tag)
            # ``HierarchyNode.activity_condition.requirements`` is already keyed by
            # indicator local index (int), so no tag→index translation is needed.
            req = (
                {int(k): int(v) for k, v in nodes[0].activity_condition.requirements.items()}
                if nodes
                else {}
            )
            activity_conditions.extend([dict(req)] * sub_dim)
        col += sub_dim

    if lowers:
        feature_bounds = tf.stack(
            [tf.concat(lowers, axis=0), tf.concat(uppers, axis=0)], axis=-1
        )
    else:
        feature_bounds = tf.zeros([0, 2], dtype=gpflow.default_float())
    return feature_dims, feature_bounds, indicator_dims, activity_conditions


feature_dims, feature_bounds, indicator_dims, activity_conditions = (
    primitives_from_space(space)
)
print("feature_dims:        ", feature_dims)
print("indicator_dims:      ", indicator_dims)
print("feature_bounds:\n", feature_bounds.numpy())
print("activity_conditions: ", activity_conditions)

# %% [markdown]
# ## A self-contained Arc kernel
#
# The Arc kernel maps each conditional feature column $c$ (normalised value
# $v_c \in [0, 1]$) into the plane via
#
# $$\phi_c(v_c, m_c) = \big( r_c \sin(\pi a_c v_c)\, m_c,\;\;
#                            r_c \cos(\pi a_c v_c)\, m_c \big),$$
#
# where $m_c \in \{0, 1\}$ is the activity mask (1 if all the column's
# indicator requirements are met by this point, 0 otherwise) and $r_c$, $a_c$
# are trainable parameters. Inactive points map to the origin; active points
# sit on a circle whose phase is a function of $v_c$. The base kernel
# evaluates covariance in the concatenated embedded vector — unconditional
# columns pass through normalised, conditional columns contribute their two
# embedded coordinates each.

# %%
_IGNORE = -1


class ArcKernel(gpflow.kernels.Kernel):
    def __init__(
        self,
        feature_dims,
        feature_bounds,
        indicator_dims=(),
        activity_conditions=(),
        base_kernel=None,
    ):
        super().__init__()

        feature_dims = list(feature_dims)
        indicator_dims = list(indicator_dims)
        activity_conditions = list(activity_conditions)
        n_feat, n_ind = len(feature_dims), len(indicator_dims)

        # Compile per-column requirements into a [D_f, D_i] int32 tensor.
        # `_IGNORE` marks "this indicator is irrelevant for this column".
        required = np.full((n_feat, n_ind), _IGNORE, dtype=np.int32)
        for j, req in enumerate(activity_conditions):
            for k, v in req.items():
                required[j, int(k)] = int(v)

        self._feature_dims = tf.constant(feature_dims, dtype=tf.int32)
        self._indicator_dims = tf.constant(indicator_dims, dtype=tf.int32)
        self._bounds = tf.convert_to_tensor(
            feature_bounds, dtype=gpflow.default_float()
        )
        self._required = tf.constant(required, dtype=tf.int32)
        self._required_is_ignore = tf.equal(self._required, _IGNORE)

        cond_set = {j for j in range(n_feat) if (required[j] != _IGNORE).any()}
        self._cond_local_idx = sorted(cond_set)
        self._uncond_local_idx = [j for j in range(n_feat) if j not in cond_set]
        self._n_cond = len(self._cond_local_idx)
        self._n_uncond = len(self._uncond_local_idx)
        self._n_feat, self._n_ind = n_feat, n_ind

        # Stationary base kernel; lengthscale frozen at 1 so the embedding
        # parameters carry the distance scale.
        if base_kernel is None:
            base_kernel = gpflow.kernels.Matern52()
        if not isinstance(base_kernel, gpflow.kernels.Stationary):
            raise ValueError("base_kernel must be a stationary GPflow kernel.")
        base_kernel.lengthscales.assign(tf.ones_like(base_kernel.lengthscales))
        gpflow.utilities.set_trainable(base_kernel.lengthscales, False)
        self.base_kernel = base_kernel

        if self._n_cond > 0:
            self.angle = gpflow.Parameter(
                0.5 * tf.ones(self._n_cond, dtype=gpflow.default_float()),
                transform=tfp.bijectors.Sigmoid(
                    to_default_float(0.1), to_default_float(0.9)
                ),
                name="angle",
            )
            self.radius = gpflow.Parameter(
                tf.ones(self._n_cond, dtype=gpflow.default_float()),
                transform=positive(),
                name="radius",
            )

    def _build_activity_mask(self, X):
        if self._n_ind == 0:
            return tf.ones([tf.shape(X)[0], self._n_feat], dtype=tf.bool)
        ind = tf.gather(X, self._indicator_dims, axis=-1)
        ind = tf.cast(tf.round(tf.cast(ind, gpflow.default_float())), tf.int32)
        match = tf.logical_or(
            self._required_is_ignore[None, :, :],
            tf.equal(ind[:, None, :], self._required[None, :, :]),
        )
        return tf.reduce_all(match, axis=-1)  # [N, D_f]

    def _normalise(self, X):
        X = tf.cast(X, gpflow.default_float())
        v = tf.gather(X, self._feature_dims, axis=-1)
        lo, hi = self._bounds[:, 0], self._bounds[:, 1]
        ranges = tf.where(
            tf.abs(hi - lo) < 1e-12, tf.ones_like(hi - lo), hi - lo
        )
        return (v - lo) / ranges

    def _embed(self, X):
        v = self._normalise(X)
        m = tf.cast(self._build_activity_mask(X), gpflow.default_float())
        parts = []
        if self._n_uncond > 0:
            parts.append(tf.gather(v, self._uncond_local_idx, axis=-1))
        if self._n_cond > 0:
            v_c = tf.gather(v, self._cond_local_idx, axis=-1)
            m_c = tf.gather(m, self._cond_local_idx, axis=-1)
            theta = np.pi * self.angle * v_c
            parts.append(self.radius * tf.sin(theta) * m_c)
            parts.append(self.radius * tf.cos(theta) * m_c)
        if not parts:
            return tf.zeros([tf.shape(X)[0], 0], dtype=gpflow.default_float())
        return tf.concat(parts, axis=-1)

    def K(self, X, X2=None):
        Z1 = self._embed(X)
        Z2 = self._embed(X2) if X2 is not None else Z1
        return self.base_kernel.K(Z1, Z2)

    def K_diag(self, X):
        return self.base_kernel.K_diag(self._embed(X))


arc = ArcKernel(
    feature_dims, feature_bounds, indicator_dims, activity_conditions
)
print("n_uncond:", arc._n_uncond, "  n_cond:", arc._n_cond)

# %% [markdown]
# ## Inspect the activity mask
#
# A quick sanity check on a hand-crafted batch. The non-indicator columns are,
# in `feature_dims` order, $x_1, x_2, x_3$. With $y_1 = 1$ we expect $x_1$ and
# $x_2$ active; with $y_1 = 0$ we expect $x_1$ and $x_3$ active.

# %%
X_demo = tf.constant(
    [
        [0.5, 1.0, 2.5, 0.0],  # y1 = 1: x1 + x2 active
        [0.5, 0.0, 2.5, 0.0],  # y1 = 0: x1 + x3 active
    ],
    dtype=tf.float64,
)
mask = arc._build_activity_mask(X_demo).numpy()
print("columns correspond to feature_dims =", feature_dims, "(i.e. x1, x2, x3)")
print(mask)

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
arc_for_fit = ArcKernel(
    feature_dims, feature_bounds, indicator_dims, activity_conditions
)
kernel = gpflow.kernels.Constant() * arc_for_fit
gpr = gpflow.models.GPR(
    data=(X_train, Y_train), kernel=kernel, noise_variance=0.05
)

print(f"LML before fit: {gpr.log_marginal_likelihood().numpy():+.3f}")
gpflow.optimizers.Scipy().minimize(
    gpr.training_loss, gpr.trainable_variables, options={"maxiter": 100}
)
print(f"LML after fit:  {gpr.log_marginal_likelihood().numpy():+.3f}")
print("learnt angle :", arc_for_fit.angle.numpy())
print("learnt radius:", arc_for_fit.radius.numpy())

# %% [markdown]
# ## Predict on held-out points
#
# We evaluate on a small held-out batch covering both branches. The kernel
# must place the two test rows on different parts of the embedded space, so
# their predicted means follow the corresponding branch's signal.

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
# ## A Wedge variant
#
# The Wedge kernel (Horn et al. 2019) replaces Arc's circle embedding with a
# triangular one:
#
# $$\phi_c(v_c, m_c) = \big(
#     (\theta_1 v_c + \theta_2 v_c \cos\rho)\, m_c,\;\;
#     (\theta_2 v_c \sin\rho)\, m_c
#   \big).$$
#
# The "incomparable" distance now scales with the active value $v_c$ rather
# than being constant in it — closer to what a practitioner expects near
# disjunction boundaries. Subclassing the Arc skeleton is a one-method change.


# %%
class WedgeKernel(ArcKernel):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        if self._n_cond > 0:
            del self.angle, self.radius
            self.theta1 = gpflow.Parameter(
                tf.ones(self._n_cond, dtype=gpflow.default_float()),
                transform=positive(),
                name="theta1",
            )
            self.theta2 = gpflow.Parameter(
                tf.ones(self._n_cond, dtype=gpflow.default_float()),
                transform=positive(),
                name="theta2",
            )
            self.rho = gpflow.Parameter(
                0.5
                * np.pi
                * tf.ones(self._n_cond, dtype=gpflow.default_float()),
                transform=tfp.bijectors.Sigmoid(
                    to_default_float(1e-6), to_default_float(np.pi)
                ),
                name="rho",
            )

    def _embed(self, X):
        v = self._normalise(X)
        m = tf.cast(self._build_activity_mask(X), gpflow.default_float())
        parts = []
        if self._n_uncond > 0:
            parts.append(tf.gather(v, self._uncond_local_idx, axis=-1))
        if self._n_cond > 0:
            v_c = tf.gather(v, self._cond_local_idx, axis=-1)
            m_c = tf.gather(m, self._cond_local_idx, axis=-1)
            comp1 = (
                self.theta1 * v_c + self.theta2 * v_c * tf.cos(self.rho)
            ) * m_c
            comp2 = (self.theta2 * v_c * tf.sin(self.rho)) * m_c
            parts.extend([comp1, comp2])
        if not parts:
            return tf.zeros([tf.shape(X)[0], 0], dtype=gpflow.default_float())
        return tf.concat(parts, axis=-1)


wedge = WedgeKernel(
    feature_dims, feature_bounds, indicator_dims, activity_conditions
)
K_wedge = wedge.K(X_demo).numpy()
print("Wedge kernel matrix on demo points:")
print(K_wedge)

# %% [markdown]
# ## What this shows
#
# The whole conditional-GP pipeline above relies only on:
#
# * `BooleanSearchSpace`, `Box`, `HierarchyNode`, `HierarchicalSearchSpace`
#   from `trieste.space`;
# * GPflow / TF / TFP for the differentiable kernel itself.
