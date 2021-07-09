# -*- coding: utf-8 -*-
# %% [markdown]
# # Multi-objective optimization with Expected HyperVolume Improvement

# %%
import math
import gpflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from util.plotting import plot_bo_points, plot_function_2d, plot_mobo_history, plot_mobo_points_in_obj_space

# %%
import trieste
from trieste.acquisition.function import ExpectedHypervolumeImprovement
from trieste.data import Dataset
from trieste.models import create_model
from trieste.models.model_interfaces import ModelStack
from trieste.space import Box
from trieste.utils.multi_objectives import VLMOP2
from trieste.utils.pareto import Pareto, get_reference_point
from trieste.acquisition.rule import EfficientGlobalOptimization

np.random.seed(1793)
tf.random.set_seed(1793)

# %% [markdown]
# ## Describe the problem
#
# In this tutorial, we provide a multi-objective optimization example with constraints.


# %%
vlmop2 = VLMOP2().objective()
class Sim:
    threshold = 0.5

    @staticmethod
    def objective(input_data):
        return vlmop2(input_data)

    @staticmethod
    def constraint(input_data):
        x, y = input_data[:, -2], input_data[:, -1]
        z = tf.cos(x) * tf.cos(y) - tf.sin(x) * tf.sin(y)
        return z[:, None]

class Sim1(Sim):
    @staticmethod
    def objective(input_data):
        return vlmop2(input_data)[:, 0:1]

class Sim2(Sim):
    @staticmethod
    def objective(input_data):
        return vlmop2(input_data)[:, 1:2]


OBJECTIVE = "OBJECTIVE"
CONSTRAINT = "CONSTRAINT"

def observer(query_points):
    return {
        OBJECTIVE: Dataset(query_points, Sim.objective(query_points)),
        CONSTRAINT: Dataset(query_points, Sim.constraint(query_points)),
    }

# %%
mins = [-2, -2]
maxs = [2, 2]
search_space = Box(mins, maxs)
num_objective = 2

# %% [markdown]
# Let's randomly sample some initial data from the observer ...

# %%
num_initial_points = 20
initial_query_points = search_space.sample(num_initial_points)
initial_data = observer(initial_query_points)

from util.inequality_constraints_utils import plot_init_query_points
plot_init_query_points(
    search_space,
    Sim1,
    initial_data[OBJECTIVE].astuple(),
    initial_data[CONSTRAINT].astuple(),
)
plt.show()

plot_init_query_points(
    search_space,
    Sim2,
    initial_data[OBJECTIVE].astuple(),
    initial_data[CONSTRAINT].astuple(),
)
plt.show()

# %% [markdown]
# ... and visualise the data across the design space: each figure contains the contour lines of each objective function.

# %%
_, ax = plot_function_2d(
    vlmop2,
    mins,
    maxs,
    grid_density=100,
    contour=True,
    title=["Obj 1", "Obj 2"],
    figsize=(12, 6),
    colorbar=True,
    xlabel="$X_1$",
    ylabel="$X_2$",
)
plot_bo_points(initial_query_points, ax=ax[0, 0], num_init=num_initial_points)
plot_bo_points(initial_query_points, ax=ax[0, 1], num_init=num_initial_points)
plt.show()

# %% [markdown]
# ... and in the objective space. The `plot_mobo_points_in_obj_space` will automatically search for non-dominated points and colours them in purple.

# %%
plot_mobo_points_in_obj_space(initial_data[OBJECTIVE].observations)
plt.show()


# %% [markdown]
# ## Modelling the two functions
#
# In this example we model the two objective functions individually with their own Gaussian process models, for problems where the objective functions are similar it may make sense to build a joint model.
#
# We use a model wrapper: `ModelStack` to stack these two independent GP into a single model working as a (independent) multi-output model.


# %%
def build_stacked_independent_objectives_model(data: Dataset, num_output) -> ModelStack:
        gprs =[]
        for idx in range(num_output):
            single_obj_data = Dataset(data.query_points, tf.gather(data.observations, [idx], axis=1))
            variance = tf.math.reduce_variance(single_obj_data.observations)
            kernel = gpflow.kernels.Matern52(variance)
            gpr = gpflow.models.GPR((single_obj_data.query_points, single_obj_data.observations), kernel, noise_variance=1e-5)
            gpflow.utilities.set_trainable(gpr.likelihood, False)
            gprs.append((create_model({
            "model": gpr,
            "optimizer": gpflow.optimizers.Scipy(),
            "optimizer_args": {
            "minimize_args": {"options": dict(maxiter=100)}}}), 1))

        return ModelStack(*gprs)


# %%
objective_model = build_stacked_independent_objectives_model(initial_data[OBJECTIVE], num_objective)

def create_constraint_model(data):
    variance = tf.math.reduce_variance(data.observations)
    lengthscale = 1.0 * np.ones(2, dtype=gpflow.default_float())
    kernel = gpflow.kernels.Matern52(variance=variance, lengthscales=lengthscale)
    jitter = gpflow.kernels.White(1e-12)
    gpr = gpflow.models.GPR(data.astuple(), kernel + jitter, noise_variance=1e-5)
    gpflow.set_trainable(gpr.likelihood, False)
    return trieste.models.create_model({
        "model": gpr,
        "optimizer": gpflow.optimizers.Scipy(),
        "optimizer_args": {
            "minimize_args": {"options": dict(maxiter=100)},
        },
    })
constraint_model = create_constraint_model(initial_data[CONSTRAINT])

model = {OBJECTIVE: objective_model, CONSTRAINT: constraint_model}

# %% [markdown]
# ## Define the acquisition function
# Here we utilize the [EHVI](https://link.springer.com/article/10.1007/s10898-019-00798-7): `ExpectedHypervolumeImprovement` acquisition function:

# %%
from trieste.acquisition.function import ExpectedConstrainedHypervolumeImprovement

pof = trieste.acquisition.ProbabilityOfFeasibility(threshold=Sim.threshold)
echvi = ExpectedConstrainedHypervolumeImprovement(OBJECTIVE, pof.using(CONSTRAINT))
rule = EfficientGlobalOptimization(builder=echvi)  # type: ignore

# %% [markdown]
# ## Run the optimization loop
#
# We can now run the optimization loop

# %%
num_steps = 30
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
result = bo.optimize(num_steps, initial_data, model, acquisition_rule=rule)

# %% [markdown]
# To conclude, we visualize the queried data across the design space.
# We represent the initial points as crosses and the points obtained by our optimization loop as dots.

# %%
objective_dataset = result.final_result.unwrap().datasets[OBJECTIVE]
constraint_dataset = result.final_result.unwrap().datasets[CONSTRAINT]
data_query_points = objective_dataset.query_points
data_observations = objective_dataset.observations

plot_init_query_points(
    search_space,
    Sim1,
    objective_dataset.astuple(),
    constraint_dataset.astuple(),
)
plt.show()

# %% [markdown]
# Visualize in objective space. Purple dots denote the non-dominated points.

# %%
plot_mobo_points_in_obj_space(data_observations, num_init=num_initial_points)
plt.show()
