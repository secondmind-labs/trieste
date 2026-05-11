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
# # Arc and Wedge kernels with constraints on a `HierarchicalSearchSpace`
#
# Two earlier notebooks each cover half of the constrained-conditional-BO
# story:
#
# * `hierarchical_search_space.pct.py` introduces `HierarchicalSearchSpace`,
#   `HierarchyNode`, and the indicator-centric API for describing a
#   Generalized Disjunctive Program.
# * `hierarchical_kernel_from_scratch.pct.py` builds Arc and Wedge kernels
#   from scratch and fits a GP on a synthetic disjunctive objective —
#   without constraints.
#
# This notebook wires the two together. It attaches three constraint
# families — a *global* constraint, an indicator-gated *conditional*
# constraint, and a *logical proposition* on the indicators — to the same
# four-variable disjunction; redefines the Arc and Wedge kernels inline so
# the notebook stays self-contained; and closes the loop with a short
# feasibility-filtered Bayesian-optimisation loop that combines the GP
# surrogate with `space.is_feasible` at the acquisition step.

# %%
import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.utilities import positive, to_default_float

from trieste.space import (
    BooleanSearchSpace,
    Box,
    ConditionalConstraint,
    HierarchicalSearchSpace,
    HierarchyNode,
    INACTIVE_CONSTRAINT_RESIDUAL,
    LinearConstraint,
    LogicalProposition,
)

np.random.seed(1793)
tf.random.set_seed(1793)

# %% [markdown]
# ## Build the search space (unconstrained baseline)
#
# Six-variable disjunction with two Boolean indicators:
#
# * $x_1$ — continuous, unconditional;
# * $y_1$ — Boolean indicator;
# * $y_2$ — Boolean indicator;
# * $x_2$ — continuous, active when $y_1 = 1$;
# * $x_3$ — continuous, active when $y_1 = 0$;
# * $x_4$ — continuous, active when $y_2 = 1$.
#
# `HierarchicalSearchSpace` validation rejects an indicator that does not
# gate any node, so every declared indicator must appear in at least one
# `HierarchyNode.indicator_conditions`. We give $y_2$ its own conditional
# subspace $x_4$.

# %%
spaces = [
    Box([0.0], [1.0]),  # x1: unconditional
    BooleanSearchSpace(),  # y1: Boolean indicator
    BooleanSearchSpace(),  # y2: Boolean indicator
    Box([0.0], [5.0]),  # x2: active when y1 = 1
    Box([-1.0], [1.0]),  # x3: active when y1 = 0
    Box([0.0], [2.0]),  # x4: active when y2 = 1
]
tags = ["x1", "y1", "y2", "x2", "x3", "x4"]
hierarchy = [
    HierarchyNode("shared", subspace_tags=["x1"], indicator_conditions={}),
    HierarchyNode(
        "branch_A", subspace_tags=["x2"], indicator_conditions={"y1": True}
    ),
    HierarchyNode(
        "branch_B", subspace_tags=["x3"], indicator_conditions={"y1": False}
    ),
    HierarchyNode(
        "branch_C", subspace_tags=["x4"], indicator_conditions={"y2": True}
    ),
]
space_unconstrained = HierarchicalSearchSpace(
    spaces, tags, hierarchy, indicator_tags=["y1", "y2"]
)
print("dimension:        ", int(space_unconstrained.dimension))
print("indicator_tags:   ", space_unconstrained.indicator_tags)
print("non_indicator_tags:", space_unconstrained.non_indicator_tags)
print("has_constraints:  ", space_unconstrained.has_constraints)

# %% [markdown]
# Flat-vector layout is `[x1, y1, y2, x2, x3]` in `tags` order. `y2` does not
# gate any non-indicator subspace; it exists only to participate in a
# constraint and a logical proposition. The hierarchy itself is independent
# of constraints — the indicator can be unused by the hierarchy and still
# carry feasibility information.

# %% [markdown]
# ## Attach a global linear constraint
#
# A *global* constraint is enforced on every point regardless of indicator
# values. The trieste primitive is `LinearConstraint(A, lb, ub)`, where `A`
# is a `[C, D]` matrix evaluated on the full flat vector. Here we enforce
#
# $$x_1 + 0.5 \, x_2 \;\le\; 0.9$$
#
# with `A = [[1, 0, 0, 0.5, 0, 0]]` (zeros on `y1`, `y2`, `x3`, `x4`),
# `lb = -inf`, `ub = 0.9`.

# %%
global_constraint = LinearConstraint(
    A=np.array([[1.0, 0.0, 0.0, 0.5, 0.0, 0.0]]),
    lb=np.array([-np.inf]),
    ub=np.array([0.9]),
)
space_global = HierarchicalSearchSpace(
    spaces,
    tags,
    hierarchy,
    indicator_tags=["y1", "y2"],
    global_constraints=[global_constraint],
)
print("has_constraints:", space_global.has_constraints)

samples = space_global.sample(16)
feasible_mask = space_global.is_feasible(samples).numpy()
print(f"feasible: {feasible_mask.sum()}/16")
print("first 4 samples (with feasibility flag):")
for x, ok in zip(samples.numpy()[:4], feasible_mask[:4]):
    print(f"  {x}  feasible={bool(ok)}")

# %% [markdown]
# `HierarchicalSearchSpace.sample` does **not** rejection-sample for
# feasibility (that is what `sample_feasible` is for). The plain `sample`
# call returns points uniformly from the bounding box, and we filter
# downstream with `is_feasible`. This separation is deliberate: the user
# chooses whether to pay the rejection-sampling cost.

# %% [markdown]
# ## Attach an indicator-gated conditional constraint
#
# A *conditional* constraint is enforced only when its indicator conditions
# are met. We add $x_3 \ge -0.5$ that is enforced when $y_1 = 0$ (the only
# branch in which $x_3$ is semantically meaningful):

# %%
conditional_constraint = ConditionalConstraint(
    constraint=LinearConstraint(
        A=np.array([[1.0]]),
        lb=np.array([-0.5]),
        ub=np.array([np.inf]),
    ),
    indicator_conditions={"y1": False},
    active_subspace_tags=["x3"],
)
space_with_cond = HierarchicalSearchSpace(
    spaces,
    tags,
    hierarchy,
    indicator_tags=["y1", "y2"],
    global_constraints=[global_constraint],
    conditional_constraints=[conditional_constraint],
)

# %% [markdown]
# `constraints_residuals` returns the per-point residual stacked across
# constraint sources. To see the *Big-M inactive-residual* behaviour clearly
# we hand-craft a batch with one row per branch:

# %%
demo_batch = tf.constant(
    [
        [0.30, 1.0, 0.0, 1.50, 0.00, 0.00],  # y1=1: x3 inactive  -> big-M
        [0.30, 0.0, 0.0, 0.00, -0.20, 0.00],  # y1=0: x3 active and feasible
        [0.30, 0.0, 0.0, 0.00, -0.80, 0.00],  # y1=0: x3 active and violates
    ],
    dtype=tf.float64,
)
residuals = space_with_cond.constraints_residuals(demo_batch).numpy()
print(
    "residuals per row (global lb, global ub, conditional lb, conditional ub):"
)
for row, r in zip(demo_batch.numpy(), residuals):
    print(f"  {row}  ->  {r}")
print(f"\nINACTIVE_CONSTRAINT_RESIDUAL = {INACTIVE_CONSTRAINT_RESIDUAL:.1e}")

# %% [markdown]
# Notice that the first row carries the $10^{10}$ Big-M sentinel in the two
# conditional columns: the constraint is inactive when $y_1 = 1$, so the
# point is trivially feasible w.r.t. it. The second row's conditional
# residual is positive (feasible by a margin of $0.3$); the third row's is
# negative (infeasible by $0.3$). This is exactly the GDP Big-M
# reformulation trick — gradient-based solvers handle the indicator
# discontinuity by never crossing it during a single polish step.

# %% [markdown]
# ## Attach a logical proposition
#
# A *logical proposition* is a constraint over the indicators only. It has
# no gradient and is therefore excluded from `constraints_residuals`, but
# `is_feasible` enforces it.
#
# Example: "if $y_2 = 1$, then $y_1 = 1$" — i.e. $y_2 = 1$ implies $y_1 = 1$.

# %%
y2_implies_y1 = LogicalProposition(
    fun=lambda ind: tf.logical_or(
        tf.equal(tf.cast(ind["y2"], tf.int32), 0),
        tf.equal(tf.cast(ind["y1"], tf.int32), 1),
    )[:, 0],
    name="y2_implies_y1",
)
space = HierarchicalSearchSpace(
    spaces,
    tags,
    hierarchy,
    indicator_tags=["y1", "y2"],
    global_constraints=[global_constraint],
    conditional_constraints=[conditional_constraint],
    logical_propositions=[y2_implies_y1],
)
print("has_constraints:", space.has_constraints)

# Logical-proposition check: a point with y2=1 and y1=0 must fail is_feasible.
probe = tf.constant(
    [
        [0.10, 0.0, 1.0, 0.0, 0.0, 0.5],  # y2=1, y1=0 -> violates proposition
        [0.10, 1.0, 1.0, 1.0, 0.0, 0.5],  # y2=1, y1=1 -> satisfies proposition
        [0.10, 0.0, 0.0, 0.0, 0.0, 0.0],  # y2=0 -> proposition vacuously true
    ],
    dtype=tf.float64,
)
print("is_feasible:", space.is_feasible(probe).numpy())

# %% [markdown]
# Crucially, `constraints_residuals` is still well-defined on these points
# — it returns finite values for the gradient-compatible sources (global +
# conditional) and never sees the proposition:

# %%
print("constraints_residuals (no proposition contribution):")
print(space.constraints_residuals(probe).numpy())

# %% [markdown]
# This split — gradient-compatible residuals vs.\ indicator-only propositions
# — is what lets a gradient-based polish step in 2 step GA + gradient based optimisers
# work unchanged: the polish fixes the indicators, so the proposition is constant
# over the polish trajectory and can be filtered upstream.

# %% [markdown]
# ## Translate the search space into kernel primitives
#
# The kernel doesn't need to know about constraints, only about the
# disjunctive structure. The translation from a `HierarchicalSearchSpace`
# to integer-indexed kernel primitives is the same as in
# `hierarchical_kernel_from_scratch.pct.py` — repeated here verbatim so
# this notebook stays self-contained.


# %%
def primitives_from_space(space):
    indicator_set = set(space.indicator_tags)
    indicator_local_by_tag = {t: k for k, t in enumerate(space.indicator_tags)}

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
            req = (
                {
                    indicator_local_by_tag[t]: int(v)
                    for t, v in nodes[0].indicator_conditions.items()
                }
                if nodes and nodes[0].indicator_conditions
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
# Note that `y2` appears in `indicator_dims` even though no
# `HierarchyNode.indicator_conditions` references it — it's part of the
# flat vector, the kernel sees its column, and the activity mask is
# computed over both indicators (with no requirement on `y2` for any
# feature). Constraints know about `y2`; the kernel is indifferent.

# %% [markdown]
# ## A self-contained Arc kernel
#
# The Arc kernel embeds each conditional feature column $c$ (normalised
# value $v_c \in [0, 1]$) into the plane via
#
# $$\phi_c(v_c, m_c) = \big( r_c \sin(\pi a_c v_c)\, m_c,\;\;
#                            r_c \cos(\pi a_c v_c)\, m_c \big),$$
#
# where $m_c \in \{0, 1\}$ is the activity mask and $r_c, a_c$ are
# trainable per-dimension parameters. Inactive points sit at the origin;
# active points lie on a circle of radius $r_c$ at an angle that depends
# on $v_c$. The base kernel evaluates covariance in the concatenated
# embedded vector.

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
# ## A Wedge variant
#
# Same embedding skeleton; the conditional features map onto a wedge
# rather than a circle. Subclassing is a one-method change.


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


# %% [markdown]
# ## Synthetic disjunctive objective and feasibility-filtered training set
#
# Same objective as the predecessor notebook:
#
# $$f(x_1, y_1, y_2, x_2, x_3, x_4) =
#     \sin(2\pi x_1)
#   \;+\; \mathbf{1}[y_1 = 1] \cdot \tfrac{1}{2} \cos(\pi x_2 / 5)
#   \;+\; \mathbf{1}[y_1 = 0] \cdot \tfrac{1}{2} x_3
#   \;+\; \mathbf{1}[y_2 = 1] \cdot \tfrac{3}{10} x_4.$$
#
# We draw 80 raw samples and filter to feasible points.


# %%
def objective(X):
    X = np.asarray(X)
    x1 = X[:, 0]
    y1 = X[:, 1]
    y2 = X[:, 2]
    x2 = X[:, 3]
    x3 = X[:, 4]
    x4 = X[:, 5]
    return (
        np.sin(2.0 * np.pi * x1)
        + (y1 > 0.5).astype(float) * 0.5 * np.cos(np.pi * x2 / 5.0)
        + (y1 < 0.5).astype(float) * 0.5 * x3
        + (y2 > 0.5).astype(float) * 0.3 * x4
    ).reshape(-1, 1)


raw = space.sample(80).numpy()
mask = space.is_feasible(raw).numpy()
X_train = raw[mask]
Y_train = objective(X_train) + 0.05 * np.random.randn(X_train.shape[0], 1)
print(f"feasibility-filter: kept {X_train.shape[0]}/80 raw samples")

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
# ## Feasibility-filtered Bayesian-optimisation loop
#
# A short ask-tell loop. Each iteration:
#
# 1. Draw a pool of 200 candidates with `space.sample`.
# 2. Filter to feasible candidates via `space.is_feasible`.
# 3. Score the feasible candidates with Expected Improvement against the
#    current GP posterior:
#    $\mathrm{EI}(x) = (f^\star - \mu)\,\Phi(z) + \sigma\,\phi(z)$,
#    where $z = (f^\star - \mu)/\sigma$ and $f^\star$ is the running best
#    feasible observation.
# 4. Evaluate the highest-EI feasible candidate.
# 5. Refit the GP from scratch with a fresh kernel.
#
# Five steps is illustrative, not exhaustive — the point is to show that
# `is_feasible` slots cleanly into the acquisition stage, not to claim
# convergence on a 4-D disjunctive problem.


# %%
def expected_improvement(model, candidates, best):
    mean, var = model.predict_f(candidates)
    mean = tf.reshape(mean, [-1])
    sigma = tf.sqrt(tf.maximum(tf.reshape(var, [-1]), 1e-12))
    normal = tfp.distributions.Normal(
        loc=tf.zeros_like(mean), scale=tf.ones_like(mean)
    )
    z = (best - mean) / sigma
    ei = (best - mean) * normal.cdf(z) + sigma * normal.prob(z)
    return tf.maximum(ei, 0.0)


def refit_arc(X, Y):
    arc_local = ArcKernel(
        feature_dims, feature_bounds, indicator_dims, activity_conditions
    )
    kernel_local = gpflow.kernels.Constant() * arc_local
    gpr_local = gpflow.models.GPR(
        data=(X, Y), kernel=kernel_local, noise_variance=0.05
    )
    gpflow.optimizers.Scipy().minimize(
        gpr_local.training_loss,
        gpr_local.trainable_variables,
        options={"maxiter": 100},
    )
    return gpr_local, arc_local


n_bo_steps = 5
best_so_far = float(Y_train.min())
print(f"initial best feasible observation: {best_so_far:+.3f}")
for step in range(n_bo_steps):
    pool = space.sample(200).numpy()
    feasible = pool[space.is_feasible(pool).numpy()]
    if feasible.shape[0] == 0:
        print(f"step {step+1}: no feasible candidates in pool; skipping")
        continue
    ei = expected_improvement(
        gpr, tf.constant(feasible, dtype=tf.float64), best_so_far
    ).numpy()
    idx = int(np.argmax(ei))
    x_next = feasible[idx : idx + 1]
    y_next = objective(x_next) + 0.05 * np.random.randn(1, 1)
    X_train = np.concatenate([X_train, x_next], axis=0)
    Y_train = np.concatenate([Y_train, y_next], axis=0)
    best_so_far = float(Y_train.min())
    gpr, arc_for_fit = refit_arc(X_train, Y_train)
    print(
        f"step {step+1}: picked x = {x_next.ravel().tolist()}  "
        f"y = {y_next.item():+.3f}  best = {best_so_far:+.3f}  "
        f"feasible-pool = {feasible.shape[0]}/200"
    )

# %% [markdown]
# Two things are worth noticing from the loop output:
#
# * The feasible-pool size each iteration is well below 200 — the global
#   linear constraint, the indicator-gated $x_3 \ge -0.5$, and the
#   $y_2 \Rightarrow y_1$ proposition together carve out a non-trivial
#   feasible region.
# * The GP is refit from scratch each iteration; the per-iteration cost
#   could be reduced by warm-starting the parameters, but the loop is
#   already fast enough to keep the notebook example readable.

# %% [markdown]
# ## Predict on a held-out batch (mix feasible and infeasible)
#
# The kernel is feasibility-agnostic — it predicts on infeasible points
# too. We include one infeasible row to make the point.

# %%
X_test = np.array(
    [
        [0.30, 1.0, 0.0, 1.00, 0.00, 0.00],  # y1=1, feasible
        [0.30, 0.0, 0.0, 0.00, 0.50, 0.00],  # y1=0, feasible
        [0.70, 1.0, 0.0, 0.40, 0.00, 0.00],  # y1=1, x1 high so x2 must be low
        [0.30, 0.0, 1.0, 0.00, 0.50, 1.00],  # y2=1, y1=0 -> proposition fails
    ]
)
feas = space.is_feasible(tf.constant(X_test, dtype=tf.float64)).numpy()
mean, var = gpr.predict_f(X_test)
truth = objective(X_test).ravel()
print("test predictions:")
for x, m, v, t, ok in zip(
    X_test, mean.numpy().ravel(), var.numpy().ravel(), truth, feas
):
    print(
        f"  x = {x.tolist()}  ->  mean = {m:+.3f}  var = {v:.3f}  "
        f"truth = {t:+.3f}  feasible = {bool(ok)}"
    )

# %% [markdown]
# ## Repeat with the Wedge kernel
#
# Swapping the kernel family doesn't touch the constraint plumbing or the
# BO loop at all. Two cells suffice.

# %%
wedge_for_fit = WedgeKernel(
    feature_dims, feature_bounds, indicator_dims, activity_conditions
)
kernel_w = gpflow.kernels.Constant() * wedge_for_fit
gpr_w = gpflow.models.GPR(
    data=(X_train, Y_train), kernel=kernel_w, noise_variance=0.05
)
gpflow.optimizers.Scipy().minimize(
    gpr_w.training_loss, gpr_w.trainable_variables, options={"maxiter": 100}
)
print(f"Wedge LML after fit: {gpr_w.log_marginal_likelihood().numpy():+.3f}")
print("learnt theta1:", wedge_for_fit.theta1.numpy())
print("learnt theta2:", wedge_for_fit.theta2.numpy())
print("learnt rho   :", wedge_for_fit.rho.numpy())

mean_w, _ = gpr_w.predict_f(X_test)
print("\nWedge predictions on the same held-out batch:")
for x, m_a, m_w, t in zip(
    X_test, mean.numpy().ravel(), mean_w.numpy().ravel(), truth
):
    print(
        f"  x = {x.tolist()}  ->  arc = {m_a:+.3f}  "
        f"wedge = {m_w:+.3f}  truth = {t:+.3f}"
    )

# %% [markdown]
# ## What this shows
#
# * constraint primitives (`global_constraints`,
#   `conditional_constraints`, `logical_propositions`) compose with
#   Arc and Wedge kernels with **no coupling** — the kernel never inspects
#   the constraints, the constraints never inspect the kernel.
# * The constraint API is gradient-aware where it can be (global +
#   conditional via `constraints_residuals`) and gradient-agnostic where it
#   can't be (logical propositions via `is_feasible`). 
# * The Big-M inactive-residual handles indicator-gated constraints in a
#   way gradient-based polishers can swallow without modification: a
#   conditional constraint with its indicators in the "off" state reports
#   $10^{10}$, which is feasible by an enormous margin, so the polisher
#   leaves it alone.
# * Feasibility filtering at the acquisition stage is one line
#   (`feasible = pool[space.is_feasible(pool).numpy()]`).
