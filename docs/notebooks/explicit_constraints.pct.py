# %% [markdown]
# # Explicit Constraints

# %% [markdown]
# This notebook demonstrates ways to perfom Bayesian optimization with Trieste in the presence of explicit input constraints.

# %%
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import trieste

np.random.seed(1234)
tf.random.set_seed(1234)

# %% [markdown]
# ## Describe the problem
#
# In this example, we consider the same problem presented in our [EI notebook](expected_improvement.ipynb), i.e. seeking the minimizer of the two-dimensional Branin function, but with input constraints.
#
# There are 3 linear constraints with respective lower/upper limits (i.e. 6 linear inequality constraints). There are 2 non-linear constraints with respective lower/upper limits (i.e. 4 non-linear inequality constraints).
#
# We begin our optimization after collecting five function evaluations from random locations in the search space.

# %%
from trieste.acquisition.function import fast_constraints_feasibility
from trieste.objectives import ConstrainedScaledBranin
from trieste.objectives.utils import mk_observer

observer = mk_observer(ConstrainedScaledBranin.objective)
search_space = ConstrainedScaledBranin.search_space

num_initial_points = 5
initial_query_points = search_space.sample(num_initial_points)
initial_data = observer(initial_query_points)

# %% [markdown]
# We wrap the objective and constraint functions as methods on the `Sim` class. This provides us one way to visualise the objective function, as well as the constrained objective. We get the constrained objective by masking out regions where the constraint function is above the threshold.

# %%
from trieste.experimental.plotting import plot_objective_and_constraints


class Sim:
    threshold = 0.5

    @staticmethod
    def objective(input_data):
        return ConstrainedScaledBranin.objective(input_data)

    @staticmethod
    def constraint(input_data):
        # `fast_constraints_feasibility` returns the feasibility, so we invert it. The plotting
        # function masks out values above the threshold.
        return 1.0 - fast_constraints_feasibility(search_space)(input_data)


plot_objective_and_constraints(search_space, Sim)
plt.show()

# %% [markdown]
# In addition to the normal sampling methods, the search space provides sampling methods that return feasible points only. Here we demonstrate sampling 200 feasible points from the Halton sequence.
# We can visualise the sampled points along with the objective function and the constraints.
# %%
from trieste.experimental.plotting import plot_function_2d

[xi, xj] = np.meshgrid(
    np.linspace(search_space.lower[0], search_space.upper[0], 100),
    np.linspace(search_space.lower[1], search_space.upper[1], 100),
)
xplot = np.vstack((xi.ravel(), xj.ravel())).T  # Change our input grid to list of coordinates.
constraint_values = np.reshape(search_space.is_feasible(xplot), xi.shape)

_, ax = plot_function_2d(
    ConstrainedScaledBranin.objective,
    search_space.lower,
    search_space.upper,
    grid_density=30,
    contour=True,
)

points = search_space.sample_halton_feasible(200)

ax[0, 0].scatter(points[:, 0], points[:, 1], s=15, c="tab:orange", edgecolors="black", marker="o")

ax[0, 0].contourf(
    xi, xj, constraint_values, levels=1, colors=[(0.2, 0.2, 0.2, 0.7), (1, 1, 1, 0)], zorder=1
)

ax[0, 0].set_xlabel(r"$x_1$")
ax[0, 0].set_ylabel(r"$x_2$")
plt.show()

# %% [markdown]
# ## Surrogate model
#
# We fit a surrogate Gaussian process model to the initial data. The GPflow models cannot be used directly in our Bayesian optimization routines, so we build a GPflow's `GPR` model using Trieste's convenient model build function `build_gpr` and pass it to the `GaussianProcessRegression` wrapper. Note that we set the likelihood variance to a small number because we are dealing with a noise-free problem.

# %%
from trieste.models.gpflow import GaussianProcessRegression, build_gpr

gpflow_model = build_gpr(initial_data, search_space, likelihood_variance=1e-7)
model = GaussianProcessRegression(gpflow_model)


# %% [markdown]
# ## Constrained optimization method
#
# ### Acquisition function (constrained optimization)
#
# We can construct the _expected improvement_ acquisition function as usual. In order to handle the constraints, the search space must be passed as a constructor argument.

# %%
from trieste.acquisition.function import ExpectedImprovement
from trieste.acquisition.rule import EfficientGlobalOptimization

ei = ExpectedImprovement(search_space)
rule = EfficientGlobalOptimization(ei)  # type: ignore

# %% [markdown]
# ### Run the optimization loop (constrained optimization)
#
# We can now run the optimization loop. As the search space contains constraints, the optimizer will automatically switch to using _scipy_ _trust-constr_ method to optimize the acquisition function.

# %%
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

result = bo.optimize(15, initial_data, model, acquisition_rule=rule)

# %% [markdown]
# We can now get the best point found by the optimizer. Note this isn’t necessarily the point that was last evaluated.

# %%
query_point, observation, arg_min_idx = result.try_get_optimal_point()

print(f"query point: {query_point}")
print(f"observation: {observation}")

# %% [markdown]
# We obtain the final objective and constraint data using `.try_get_final_datasets()`. We can visualise how the optimizer performed by plotting all the acquired observations, along with the true function values and optima.
#
# The crosses are the 5 initial points that were sampled from the entire search space. The green circles are the acquired observations by the optimizer. The purple circle is the best point found.

# %%
from trieste.experimental.plotting import plot_bo_points, plot_function_2d


def plot_bo_results():
    dataset = result.try_get_final_dataset()
    query_points = dataset.query_points.numpy()
    observations = dataset.observations.numpy()

    _, ax = plot_function_2d(
        ConstrainedScaledBranin.objective,
        search_space.lower,
        search_space.upper,
        grid_density=30,
        contour=True,
        figsize=(8, 6),
    )

    plot_bo_points(
        query_points, ax[0, 0], num_initial_points, arg_min_idx, c_pass="green", c_best="purple"
    )

    ax[0, 0].contourf(
        xi, xj, constraint_values, levels=1, colors=[(0.2, 0.2, 0.2, 0.7), (1, 1, 1, 0)], zorder=2
    )

    ax[0, 0].set_xlabel(r"$x_1$")
    ax[0, 0].set_ylabel(r"$x_2$")
    plt.show()


plot_bo_results()

# %% [markdown]
# ## Penalty method
#
# ### Acquisition function (penalty method)
#
# An alternative to using a constrained optimization method is to construct the _expected constrained improvement_ acquisition function similar to the [inequality-constraints notebook](inequality_constraints.ipynb). However, instead of using probability of feasibility with respect to the constraint model, we construct feasibility from the explicit input constraints. Feasibility is calculated by passing all the constraints residuals (to their respective limits) through a smoothing function and taking the product.
#
# For this method only the `FastConstraintsFeasibility` class should be passed constraints via the search space. `ExpectedConstrainedImprovement` and `BayesianOptimizer` should be set up in the normal way without constraints.
#
# Note this method penalises the expected improvement acquisition outside the feasible region. The optimizer uses the default _scipy_ _L-BFGS_ method to find the max of the acquistion function.

# %%
from trieste.acquisition.function import ExpectedConstrainedImprovement, FastConstraintsFeasibility
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.observer import OBJECTIVE

feas = FastConstraintsFeasibility(search_space)  # Search space with constraints.
eci = ExpectedConstrainedImprovement(OBJECTIVE, feas.using(OBJECTIVE))

rule = EfficientGlobalOptimization(eci)

# %% [markdown]
# ### Run the optimization loop (penalty method)
#
# We can now run the optimization loop.

# %%
# Note: use the search space without constraints for the penalty method.
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, ConstrainedScaledBranin.search_space)

result = bo.optimize(15, initial_data, model, acquisition_rule=rule)

# %% [markdown]
# We can now get the best point found by the optimizer as before.

# %%
query_point, observation, arg_min_idx = result.try_get_optimal_point()

print(f"query point: {query_point}")
print(f"observation: {observation}")

# %% [markdown]
# Plot the results as before.

# %%
plot_bo_results()

# %% [markdown]
# ## LICENSE
#
# [Apache License 2.0](https://github.com/secondmind-labs/trieste/blob/develop/LICENSE)
