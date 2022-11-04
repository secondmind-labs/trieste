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
def nlc_func(x):
    # 5-sphere.
    return tf.math.reduce_sum((x[..., :3]-.4)**2 + 2*(x[..., 3:]-.6)**2, axis=-1) - .5

nonlinear_constraints = [NonlinearConstraint(nlc_func, -1., 0.)]
nonlinear_constraints = [process_nonlinear_constraint(nlc, search_space) for nlc in nonlinear_constraints]

bound_constraints = [LinearConstraint(A=np.eye(len(search_space.lower)), lb=search_space.lower, ub=search_space.upper)]


# %% [markdown]
# ## Optimization runs

# %%
def initial_query_points(num_points=125):
    points = search_space.sample(num_points)
    # FIXME: always have at least some points in the feasible region, to avoid optimisation for constraints.
    #points = tf.concat([points, Box(lb, ub).sample(25)], axis=0)
    return points

#num_initial_samples = 200
#num_optimization_runs = 100
num_initial_samples = tf.maximum(NUM_SAMPLES_MIN, NUM_SAMPLES_DIM * tf.shape(search_space.lower)[-1])
num_optimization_runs = NUM_RUNS_DIM * tf.shape(search_space.lower)[-1]

print(f'Approx feasible region: {np.count_nonzero(constraints_satisfied(nonlinear_constraints, initial_query_points(100000))) * 100/100000}%')

# %%
## Run a dummy COBYLA optimization. The first tends to fail, for seemingly an internal optimizer issue.
#run_dummy = Run(search_space, observer, nonlinear_constraints)
#optims = {
#    "COBYLA-EI":     dict(method="COBYLA", jac=None, bounds=None, constraints=constraints_to_dict(nonlinear_constraints+bound_constraints, search_space)),
#}
#run_dummy.add_optims(optims)
#multi_run(run_dummy, 1, 1, initial_query_points, num_initial_samples=num_initial_samples, num_optimization_runs=num_optimization_runs, with_plot=False)
#
#clear_output()

# %% [markdown]
# ### Unmodified acquisition function

# %%
run_unmod = Run(search_space, observer, nonlinear_constraints)
optims = {
    "L-BFGS-EI":     None,
    "TrstRegion-EI": dict(method="trust-constr", constraints=nonlinear_constraints),
    "SLSQP-EI":      dict(method="SLSQP", constraints=constraints_to_dict(nonlinear_constraints, search_space)),
    #"COBYLA-EI":     dict(method="COBYLA", jac=None, bounds=None, constraints=constraints_to_dict(nonlinear_constraints+bound_constraints, search_space)),
}
run_unmod.add_optims(optims)

multi_run(run_unmod, 5, 5, initial_query_points, num_initial_samples=num_initial_samples, num_optimization_runs=num_optimization_runs, with_plot=False)

# %%
run_unmod.print_results_summary()
run_unmod.print_results_full()

# %% [markdown]
# ### Constrained acquistion function

# %%
run_constr = Run(search_space, observer, nonlinear_constraints, constrained_ei_type=ExpectedConstrainedImprovement, builder_kwargs=dict(min_feasibility_probability=0.5))
optims = {
    "L-BFGS-EI":     None,
}
run_constr.add_optims(optims)

multi_run(run_constr, 5, 5, initial_query_points, num_initial_samples=num_initial_samples, num_optimization_runs=num_optimization_runs, with_plot=False)

# %%
run_constr.print_results_summary()

# %% [markdown]
# ### Simple constrained acquisition function

# %%
run_simple_constr = Run(search_space, observer, nonlinear_constraints, constrained_ei_type=ExpectedSimpleConstrainedImprovement, builder_kwargs=dict(min_feasibility_probability=0.5))
optims = {
    "L-BFGS-EI":     None,
    "TrstRegion-EI": dict(method="trust-constr", constraints=nonlinear_constraints),
    "SLSQP-EI":      dict(method="SLSQP", constraints=constraints_to_dict(nonlinear_constraints, search_space)),
    #"COBYLA-EI":     dict(method="COBYLA", jac=None, bounds=None, constraints=constraints_to_dict(nonlinear_constraints+bound_constraints, search_space)),
}
run_simple_constr.add_optims(optims)

multi_run(run_simple_constr, 5, 5, initial_query_points, num_initial_samples=num_initial_samples, num_optimization_runs=num_optimization_runs, with_plot=False)

# %%
run_simple_constr.print_results_summary()

# %%
run_dummy = Run(search_space, observer, nonlinear_constraints, constrained_ei_type=ExpectedSimpleConstrainedImprovement, builder_kwargs=dict(min_feasibility_probability=0.5))
optims = {
    "SLSQP-EI":      dict(method="SLSQP", constraints=dict(type='ineq', fun=lambda x: -nlc_func(x))),
}
run_dummy.add_optims(optims)

multi_run(run_dummy, 2, 2, initial_query_points, num_initial_samples=num_initial_samples, num_optimization_runs=num_optimization_runs, with_plot=False)

run_dummy.print_results_summary()

# %% [markdown]
# ## LICENSE
#
# [Apache License 2.0](https://github.com/secondmind-labs/trieste/blob/develop/LICENSE)
