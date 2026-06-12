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
# # Constraints on a `HierarchicalSearchSpace`
#
# The previous notebooks built a `HierarchicalSearchSpace` and a GPflow
# `ArcHierarchical` kernel over it. Real disjunctive-design problems usually also
# carry **constraints**. Trieste's `HierarchicalSearchSpace` supports three kinds,
# all reusing the standard search-space contract (`constraints_residuals` /
# `is_feasible`), so a constrained space drops straight into the usual Bayesian
# optimization machinery:
#
# * **Global constraints** — ordinary `LinearConstraint` / `NonlinearConstraint`
#   objects enforced on the full flat vector (the same `constraints=` argument as
#   `Box`).
# * **Conditional (disjunctive) constraints** — `ConditionalConstraint`, enforced
#   only when a set of indicators take specified values. On inactive rows the
#   residual is the big-M sentinel `INACTIVE_CONSTRAINT_RESIDUAL`, so the point is
#   trivially feasible there and gradient-based polish never crosses the indicator
#   discontinuity.
# * **Logical propositions** — `LogicalProposition`, a boolean condition on the
#   indicators alone (no gradient); checked in `is_feasible` only.

# %%
import gpflow
import numpy as np
import tensorflow as tf
from gpflow.kernels import ArcHierarchical

from trieste.acquisition.function import ExpectedImprovement
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.bayesian_optimizer import BayesianOptimizer
from trieste.models.gpflow import GaussianProcessRegression
from trieste.objectives import mk_observer
from trieste.space import (
    INACTIVE_CONSTRAINT_RESIDUAL,
    BooleanSearchSpace,
    Box,
    ConditionalConstraint,
    HierarchicalSearchSpace,
    LinearConstraint,
    LogicalProposition,
    hierarchy_node_from_tags,
)

np.random.seed(1793)
tf.random.set_seed(1793)

# %% [markdown]
# ## A constrained disjunctive space
#
# Canonical four-variable disjunction: $x_1$ unconditional, $y_1$ a Boolean
# indicator, $x_2$ active when $y_1 = 1$, $x_3$ active when $y_1 = 0$. We attach:
#
# * a **global** constraint $x_1 + 0.5\,x_2 \leq 0.9$;
# * a **conditional** constraint $x_3 \geq -0.5$, enforced only on the $y_1 = 0$
#   branch.

# %%
subspaces = {
    "x1": Box([0.0], [1.0]),
    "y1": BooleanSearchSpace(),
    "x2": Box([0.0], [5.0]),
    "x3": Box([-1.0], [1.0]),
}
hierarchy = [
    hierarchy_node_from_tags("shared", subspace_tags=["x1"], subspaces=subspaces, indicator_tags=["y1"]),
    hierarchy_node_from_tags(
        "branch_A", subspace_tags=["x2"], activity_condition_tags={"y1": 1},
        subspaces=subspaces, indicator_tags=["y1"],
    ),
    hierarchy_node_from_tags(
        "branch_B", subspace_tags=["x3"], activity_condition_tags={"y1": 0},
        subspaces=subspaces, indicator_tags=["y1"],
    ),
]

# Global: x1 + 0.5 x2 <= 0.9  (flat columns [x1, y1, x2, x3] -> A picks x1 and x2).
global_constraint = LinearConstraint(
    A=tf.constant([[1.0, 0.0, 0.5, 0.0]], dtype=tf.float64),
    lb=[-INACTIVE_CONSTRAINT_RESIDUAL],
    ub=[0.9],
)
# Conditional: x3 >= -0.5 only when y1 == 0.
conditional_constraint = ConditionalConstraint(
    constraint=LinearConstraint(
        A=tf.constant([[1.0]], dtype=tf.float64), lb=[-0.5], ub=[INACTIVE_CONSTRAINT_RESIDUAL]
    ),
    indicator_conditions={"y1": 0},
    active_subspace_tags=["x3"],
)

space = HierarchicalSearchSpace(
    subspaces,
    hierarchy,
    indicator_tags=["y1"],
    constraints=[global_constraint],
    conditional_constraints=[conditional_constraint],
)
print("has_constraints:", space.has_constraints)

# %% [markdown]
# ## Residuals and the big-M sentinel
#
# `constraints_residuals` stacks the global and conditional residual columns.
# On a row where the conditional constraint is *inactive* ($y_1 = 1$) its columns
# are the big-M sentinel — that point is feasible with respect to the conditional
# constraint regardless of $x_3$.

# %%
# x2 = 0.4 keeps the global constraint x1 + 0.5 x2 = 0.5 <= 0.9 satisfied, so the rows
# differ only in the conditional constraint.
demo = tf.constant(
    [
        [0.3, 0.0, 0.4, 0.2],   # y1=0: conditional active (x3=0.2 >= -0.5 ok) -> feasible
        [0.3, 0.0, 0.4, -0.8],  # y1=0: conditional active (x3=-0.8 violates) -> infeasible
        [0.3, 1.0, 0.4, -0.8],  # y1=1: conditional inactive -> big-M -> feasible
    ],
    dtype=tf.float64,
)
print("residuals (cols: global[lo,hi], conditional[lo,hi]):")
print(f"  (inactive sentinel = {INACTIVE_CONSTRAINT_RESIDUAL})")
print(space.constraints_residuals(demo).numpy())
print("is_feasible:", space.is_feasible(demo).numpy())

# %% [markdown]
# ## Logical propositions (indicator-only)
#
# A `LogicalProposition` constrains the indicators alone. With two indicators we
# can express "if $y_2 = 1$ then $y_1 = 1$". It is checked by `is_feasible` but,
# having no continuous gradient, is excluded from `constraints_residuals`.

# %%
subspaces_2 = {
    "x1": Box([0.0], [1.0]),
    "y1": BooleanSearchSpace(),
    "y2": BooleanSearchSpace(),
    "x2": Box([0.0], [1.0]),
}
hierarchy_2 = [
    hierarchy_node_from_tags("shared", subspace_tags=["x1"], subspaces=subspaces_2, indicator_tags=["y1", "y2"]),
    hierarchy_node_from_tags(
        "branch", subspace_tags=["x2"], activity_condition_tags={"y1": 1, "y2": 1},
        subspaces=subspaces_2, indicator_tags=["y1", "y2"],
    ),
]
space_2 = HierarchicalSearchSpace(
    subspaces_2,
    hierarchy_2,
    indicator_tags=["y1", "y2"],
    logical_propositions=[
        LogicalProposition(
            fun=lambda ind: tf.logical_or(
                tf.not_equal(ind["y2"][..., 0], 1.0), tf.equal(ind["y1"][..., 0], 1.0)
            ),
            name="y2_implies_y1",
        )
    ],
)
pts_2 = tf.constant(
    [
        [0.5, 1.0, 1.0, 0.5],  # y2=1, y1=1 -> ok
        [0.5, 0.0, 1.0, 0.5],  # y2=1, y1=0 -> violates implication
    ],
    dtype=tf.float64,
)
print("logical is_feasible:", space_2.is_feasible(pts_2).numpy())

# %% [markdown]
# ## Constrained Bayesian optimization
#
# We minimise a synthetic disjunctive objective over the constrained `space` with
# trieste's standard `EfficientGlobalOptimization` rule and `ExpectedImprovement`
# acquisition, wrapping a GPflow `ArcHierarchical` GPR in a trieste
# `GaussianProcessRegression`.
#
# Feasibility needs a little care. The generic continuous acquisition optimizer
# would *relax* the Boolean indicators to $[0, 1]$ and only pass the **global**
# constraints to the solver — it does not see the conditional/logical constraints,
# nor the discreteness of the indicators. Since this space's feasibility is a known
# boolean function (`space.is_feasible`), the idiomatic approach is a small custom
# acquisition optimizer that rejection-samples **feasible** candidates (which are
# valid by construction) and returns the one maximising the acquisition.


# %%
def objective(X):
    X = np.asarray(X)
    x1, y1, x2, x3 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    branch_a = y1.round().astype(bool)  # y1 == 1 branch (x2 active) vs y1 == 0 branch (x3 active)
    y = (
        np.sin(2.0 * np.pi * x1)
        + np.where(branch_a, 0.5 * np.cos(np.pi * x2 / 5.0), 0.5 * x3)
    ).reshape(-1, 1)
    return tf.constant(y, dtype=tf.float64)


def feasible_sample(n, max_tries=100):
    """Rejection-sample ``n`` feasible points from the constrained space."""
    kept, tries = [], 0
    while sum(len(k) for k in kept) < n:
        if tries >= max_tries:
            raise RuntimeError(f"Could not draw {n} feasible points in {max_tries} tries.")
        pool = space.sample(4 * n)
        kept.append(tf.boolean_mask(pool, space.is_feasible(pool)).numpy())
        tries += 1
    return np.concatenate(kept, axis=0)[:n]


def feasible_acquisition_optimizer(space, target_func):
    """Maximise the acquisition over a pool of rejection-sampled feasible candidates."""
    # ``EfficientGlobalOptimization`` may pass ``(function, vectorization)``; we use one point.
    target_func = target_func[0] if isinstance(target_func, tuple) else target_func
    candidates = tf.convert_to_tensor(feasible_sample(200), dtype=tf.float64)  # [N, D]
    acquisition = target_func(candidates[:, None, :])  # [N, 1]
    return candidates[int(tf.argmax(acquisition[:, 0]))][None]  # [1, D]


observer = mk_observer(objective)
active_dims = list(range(int(space.dimension)))
initial_data = observer(tf.convert_to_tensor(feasible_sample(15), dtype=tf.float64))

# Wrap a GPflow ArcHierarchical GPR as a trieste model.
kernel = gpflow.kernels.Constant() * ArcHierarchical(
    list(space.hierarchy), active_dims=active_dims
)
gpr = gpflow.models.GPR(
    (initial_data.query_points, initial_data.observations), kernel=kernel, noise_variance=0.01
)
# num_kernel_samples=0: skip the random-restart hyperparameter sampling, which doesn't support
# ArcHierarchical's vector-valued ``angle`` parameter; we just optimise from the default init.
model = GaussianProcessRegression(gpr, num_kernel_samples=0)

rule = EfficientGlobalOptimization(
    builder=ExpectedImprovement(), optimizer=feasible_acquisition_optimizer
)
result = BayesianOptimizer(observer, space).optimize(3, initial_data, model, rule)

final_data = result.try_get_final_dataset()
print(f"best feasible objective = {float(tf.reduce_min(final_data.observations)):+.3f}")

# %% [markdown]
# ## What this shows
#
# * `HierarchicalSearchSpace` carries global, conditional, and logical constraints
#   behind the standard `is_feasible` / `constraints_residuals` interface.
# * A constrained disjunctive space drops into trieste's usual machinery — an
#   `ArcHierarchical` GPR wrapped in `GaussianProcessRegression`, optimised by
#   `EfficientGlobalOptimization` with `ExpectedImprovement`.
# * Feasibility (including the discrete indicators and the conditional/logical
#   constraints) enters through a one-line custom acquisition optimizer that maximises
#   the acquisition over `space.is_feasible` candidates; the big-M residual keeps
#   inactive disjunctive branches feasible without special-casing.
