# %% [markdown]
# # Constrained Acquisition Function Optimization with Expected Improvement - Hartmann 6

# %%
import numpy as np
import tensorflow as tf
from trieste.objectives import Hartmann6

np.random.seed(5678)
tf.random.set_seed(5678)

# %% [markdown]
# ## Load common functions and classes

# %%
# %run constrained_optimization.pct.py

# %% [markdown]
# ## Describe the problem
#
# In this example, we consider the the six-dimensional Hartmann function.

# %%
observer = mk_observer(Hartmann6.objective)
search_space = Hartmann6.search_space

# %%
Hartmann6

# %%
dims = len(search_space.lower)
#lb = (np.linspace(0.2, 0.6, dims))
lb = np.array([0.4, 0.7, 0.8, 0.5, 0.5, 0.0])
ub = np.minimum(lb + np.linspace(0.2, 0.4, dims), 1.0)
display(lb)
display(ub)
lin_constraints = [LinearConstraint(A=np.eye(len(search_space.lower)), lb=lb, ub=ub)]
#lin_constraints = [LinearConstraint(A=np.eye(len(search_space.lower)), lb=search_space.lower, ub=search_space.upper)]

bound_constraints = [LinearConstraint(A=np.eye(len(search_space.lower)), lb=search_space.lower, ub=search_space.upper)]


# %% [markdown]
# ## Optimization runs

# %%
def initial_query_points():
    points = search_space.sample(100)
    # FIXME: always have at least some points in the feasible region, to avoid optimisation for constraints.
    points = tf.concat([points, Box(lb, ub).sample(25)], axis=0)
    return points

#num_initial_samples = 200
#num_optimization_runs = 100
num_initial_samples = tf.maximum(NUM_SAMPLES_MIN, NUM_SAMPLES_DIM * tf.shape(search_space.lower)[-1])
num_optimization_runs = NUM_RUNS_DIM * tf.shape(search_space.lower)[-1]

# %%
# Run a dummy COBYLA optimization. The first tends to fail, for seemingly an internal optimizer issue.
run_dummy = Run(search_space, observer, lin_constraints)
optims = {
    "COBYLA-EI":     dict(method="COBYLA", jac=None, bounds=None, constraints=lin_constraints+bound_constraints),
}
run_dummy.add_optims(optims)
multi_run(run_dummy, 1, 1, initial_query_points, num_initial_samples=num_initial_samples, num_optimization_runs=num_optimization_runs, with_plot=False)

clear_output()

# %% [markdown]
# ### Unmodified acquisition function

# %%
run_unmod = Run(search_space, observer, lin_constraints)
optims = {
    "L-BFGS-EI":     None,
    "TrstRegion-EI": dict(method="trust-constr", constraints=lin_constraints),
    "SLSQP-EI":      dict(method="SLSQP", constraints=lin_constraints),
    "COBYLA-EI":     dict(method="COBYLA", jac=None, bounds=None, constraints=lin_constraints+bound_constraints),
}
run_unmod.add_optims(optims)

multi_run(run_unmod, 5, 5, initial_query_points, num_initial_samples=num_initial_samples, num_optimization_runs=num_optimization_runs, with_plot=False)

# %%
run_unmod.print_results_summary()
run_unmod.print_results_full()

# %% [markdown]
# ### Constrained acquistion function (penalty method)

# %%
run_constr = Run(search_space, observer, lin_constraints, constrained_ei_type=ExpectedConstrainedImprovement, builder_kwargs=dict(min_feasibility_probability=0.5))
optims = {
    "L-BFGS-EI":     None,
}
run_constr.add_optims(optims)

multi_run(run_constr, 5, 5, initial_query_points, num_initial_samples=num_initial_samples, num_optimization_runs=num_optimization_runs, with_plot=False)

# %%
run_constr.print_results_summary()

# %% [markdown]
# ### Fast constrained acquisition function

# %%
run_simple_constr = Run(search_space, observer, lin_constraints, constrained_ei_type=ExpectedFastConstrainedImprovement, builder_kwargs=dict(min_feasibility_probability=0.5))
optims = {
    "L-BFGS-EI":     None,
    "TrstRegion-EI": dict(method="trust-constr", constraints=lin_constraints),
    "SLSQP-EI":      dict(method="SLSQP", constraints=lin_constraints),
    "COBYLA-EI":     dict(method="COBYLA", jac=None, bounds=None, constraints=lin_constraints+bound_constraints),
}
run_simple_constr.add_optims(optims)

multi_run(run_simple_constr, 5, 5, initial_query_points, num_initial_samples=num_initial_samples, num_optimization_runs=num_optimization_runs, with_plot=False)

# %%
run_simple_constr.print_results_summary()

# %% [markdown]
# ## LICENSE
#
# [Apache License 2.0](https://github.com/secondmind-labs/trieste/blob/develop/LICENSE)
