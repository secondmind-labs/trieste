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
#     display_name: Python 3.10.6 (conda)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Constrained Acquisition Function Optimization with Expected Improvement - Branin function

# %%
import numpy as np
import tensorflow as tf
from trieste.objectives import ScaledBranin

np.random.seed(5678)
tf.random.set_seed(5678)

# %% [markdown]
# ## Load common functions and classes

# %%
# %run constrained_optimization.pct.py

# %% [markdown]
# ## Describe the problem
#
# In this example, we consider the the two-dimensional Branin function.

# %%
observer = mk_observer(ScaledBranin.objective)
search_space = Box([0, 0], [1, 1])

# %%
#lin_constraints = [LinearConstraint(A=np.array([[-1., 1.], [-1., 1.]]), lb=search_space.lower, ub=search_space.upper),
#                   LinearConstraint(A=np.array([[1., 0.], [0., 1.]]), lb=np.array([0., .375]), ub=np.array([.8, 1.])),
#                  ]
lin_constraints = [LinearConstraint(A=np.array([[-1., 1.], [1., 0.], [0., 1.]]), lb=np.array([-.4, .5, .2]), ub=np.array([-.2, .9, .6]))]

#def nlc_func(input_data):
#    x, y = input_data[..., -2], input_data[..., -1]
#    z = tf.cos(6*x) * tf.cos(6*y) - tf.sin(6*x) * tf.sin(6*y) + tf.sin(y)
#    return z
#nlc_func = lambda x: x[..., 0] - tf.math.sin(x[..., 1])
def nlc_func(x):
    x0p, x1p = x[..., 0], x[..., 1]
    # Rotation by 45 degrees.
    theta = -np.pi/4
    x0 = x0p*np.cos(theta) - x1p*np.sin(theta)
    x1 = x0p*np.sin(theta) + x1p*np.cos(theta)
    # Ellipse.
    return (x0-.8)**2 + 2*(x1-.1)**2 - .1

nonlinear_constraints = [NonlinearConstraint(nlc_func, -1., 0.)]
nonlinear_constraints = [process_nonlinear_constraint(nlc, search_space) for nlc in nonlinear_constraints]

bound_constraints = [LinearConstraint(A=np.eye(len(search_space.lower)), lb=search_space.lower, ub=search_space.upper)]

# %%
fig = plot_function_plotly(
    ScaledBranin.objective,
    search_space.lower,
    search_space.upper,
    grid_density=20,
)
fig.update_layout(height=400, width=400)
fig.show()

# %% [markdown]
# ### Nonlinear constraint function

# %%
fig = plot_function_plotly(
    nlc_func,
    search_space.lower,
    search_space.upper,
    grid_density=20,
)
fig.update_layout(height=400, width=400)
fig.show()

# %%
_, ax = plot_function_2d(
    ScaledBranin.objective,
    search_space.lower,
    search_space.upper,
    grid_density=30,
    contour=True,
)

#########################################
[Xi, Xj] = np.meshgrid(np.linspace(search_space.lower[0], search_space.upper[0], 50), np.linspace(search_space.lower[1], search_space.upper[1], 50))
X = np.vstack((Xi.ravel(), Xj.ravel())).T  # Change our input grid to list of coordinates.
C = np.reshape(constraints_satisfied(nonlinear_constraints, X).astype(int), Xi.shape)

plt.contourf(Xi, Xj, C, levels=1, hatches=['/', None], colors=['gray', 'white'], alpha=0.8)
#########################################

# %% [markdown]
# ## Optimization runs

# %%
def initial_query_points():
    points = search_space.sample(25)
    # FIXME: always have at least some points in the feasible region, to avoid optimisation for constraints.
    #points = tf.concat([points, Box([0.5, 0.2], [0.58, 0.28]).sample(2)], axis=0)
    #points = tf.concat([points, [[0.6, 0.4]]], axis=0)
    return points


# %%
# Run a dummy COBYLA optimization. The first tends to fail, for seemingly an internal optimizer issue.
run_dummy = Run(search_space, observer, nonlinear_constraints)
optims = {
    "COBYLA-EI":     dict(method="COBYLA", jac=None, bounds=None, constraints=nonlinear_constraints+bound_constraints),
}
run_dummy.add_optims(optims)
multi_run(run_dummy, 1, 1, initial_query_points, with_plot=False)

clear_output()

# %% [markdown]
# ### Unmodified acquisition function

# %%
#run_unmod = Run(search_space, observer, nonlinear_constraints)
#optims = {
#    "L-BFGS-EI":     None,
#    "TrstRegion-EI": dict(method="trust-constr", constraints=nonlinear_constraints),
#    "SLSQP-EI":      dict(method="SLSQP", constraints=nonlinear_constraints),
#    "COBYLA-EI":     dict(method="COBYLA", jac=None, bounds=None, constraints=nonlinear_constraints+bound_constraints),
#    "pyopt-SLSQP":   dict(method="pyopt-slsqp", constraints=nonlinear_constraints),
#    "pyopt-IPOPT":   dict(method="pyopt-ipopt", constraints=nonlinear_constraints, max_iter=1000),
#    "pyopt-ALPSO":   dict(method="pyopt-alpso", constraints=nonlinear_constraints),
#}
#run_unmod.add_optims(optims)
#
#multi_run(run_unmod, 5, 5, initial_query_points)

# %%
#run_unmod.print_results_summary()
##run_unmod.print_results_full()
#run_unmod.plot_results()
##run_unmod.write_gif()

# %% [markdown]
# ### Constrained acquistion function (penalty method)

# %%
run_constr = Run(search_space, observer, nonlinear_constraints, constrained_ei_type=ExpectedConstrainedImprovement)
optims = {
    "L-BFGS-EI":     None,
}
run_constr.add_optims(optims)

multi_run(run_constr, 5, 5, initial_query_points, num_initial_samples=20, num_optimization_runs=2)

# %%
run_constr.print_results_summary()
run_constr.plot_results()
#run_constr.write_gif()

# %% [markdown]
# ### Fast constrained acquisition function

# %%
run_simple_constr = Run(search_space, observer, nonlinear_constraints, constrained_ei_type=ExpectedFastConstrainedImprovement)
optims = {
    #"L-BFGS-EI":     None,
    "TrstRegion-EI": dict(method="trust-constr", constraints=nonlinear_constraints),
    "SLSQP-EI":      dict(method="SLSQP", constraints=nonlinear_constraints),
    "COBYLA-EI":     dict(method="COBYLA", jac=None, bounds=None, constraints=nonlinear_constraints+bound_constraints),
    "pyopt-SLSQP":   dict(method="pyopt-slsqp", constraints=nonlinear_constraints),
    "pyopt-IPOPT":   dict(method="pyopt-ipopt", constraints=nonlinear_constraints, max_iter=1000),
    "pyopt-ALPSO":   dict(method="pyopt-alpso", constraints=nonlinear_constraints),
}
run_simple_constr.add_optims(optims)

multi_run(run_simple_constr, 5, 5, initial_query_points)

# %%
run_simple_constr.print_results_summary()
run_simple_constr.plot_results()
run_simple_constr.write_gif()

# %% [markdown]
# ## LICENSE
#
# [Apache License 2.0](https://github.com/secondmind-labs/trieste/blob/develop/LICENSE)
