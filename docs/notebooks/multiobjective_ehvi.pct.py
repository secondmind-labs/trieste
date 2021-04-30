# -*- coding: utf-8 -*-
# # Multi-objective optimization with Expected HyperVolume Improvement Approach

import gpflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import math
from util.plotting import plot_bo_points, plot_function_2d, plot_mo_history

# +
import trieste
from trieste.acquisition.function import ExpectedHypervolumeImprovement
from trieste.acquisition.rule import OBJECTIVE
from trieste.models import create_model
from trieste.models.model_interfaces import ModelStack
from trieste.space import Box
from trieste.data import Dataset
from trieste.utils.multi_objectives import VLMOP2
from trieste.utils.pareto import Pareto, get_reference_point

np.random.seed(1793)
tf.random.set_seed(1793)


# -

# ## Describe the problem
#
# In this tutorial, we provide multi-objective optimization example using the expected hypervolume improvement acquisition function.   
# The synthetic function: VLMOP2 is a functions with 2 outcomes. We'll start by defining the problem parameters.


vlmop2 = VLMOP2().objective()
observer = trieste.utils.objectives.mk_observer(vlmop2, OBJECTIVE)

mins = [-2, -2]
maxs = [2, 2]
search_space = Box(mins, maxs)
num_objective = 2

# Let's randomly sample some initial data from the observer ...

num_initial_points = 10
initial_query_points = search_space.sample(num_initial_points)
initial_data = observer(initial_query_points)

# ... and visualise those data in the design space.

_, ax = plot_function_2d(
    vlmop2, mins, maxs, grid_density=100, contour=True, title=["Obj 1", "Obj 2"]
)
plot_bo_points(initial_query_points, ax=ax[0, 0], num_init=num_initial_points)
plot_bo_points(initial_query_points, ax=ax[0, 1], num_init=num_initial_points)
plt.show()

# ... and in the objective space

from util.plotting import plot_bo_points_in_obj_space
plot_bo_points_in_obj_space(initial_data[OBJECTIVE].observations)
plt.show()


# ## Modelling the two functions
#
# We'll model the different objective functions individually their own Gaussian process regression models.


def create_bo_model(data, input_dim=2, l=1.0):
    variance = tf.math.reduce_variance(data.observations)
    lengthscale = l * np.ones(input_dim, dtype=gpflow.default_float())
    kernel = gpflow.kernels.Matern52(variance=variance, lengthscales=lengthscale)
    gpr = gpflow.models.GPR(data.astuple(), kernel, noise_variance=1e-5)
    gpflow.set_trainable(gpr.likelihood, False)
    return create_model(
        {
            "model": gpr,
            "optimizer": gpflow.optimizers.Scipy(),
            "optimizer_args": {
                "minimize_args": {"options": dict(maxiter=100)},
            },
        }
    )


objective_models = [
    (
        create_bo_model(
            Dataset(
                initial_data[OBJECTIVE].query_points,
                tf.gather(initial_data[OBJECTIVE].observations, [i], axis=1),
            )
        ),
        1,
    )
    for i in range(num_objective)
]

# And stack the two independent GP model in a single `ModelStack` multioutput model representing predictions through different outputs. 

models = {OBJECTIVE: ModelStack(*objective_models)}

# ## Define the acquisition process
#
# Here we utilize the [EHVI](https://link.springer.com/article/10.1007/s10898-019-00798-7): `ExpectedHypervolumeImprovement` acquisition function:

from trieste.acquisition.rule import EfficientGlobalOptimization

ehvi = ExpectedHypervolumeImprovement()
rule: EfficientGlobalOptimization[Box] = EfficientGlobalOptimization(builder=ehvi.using(OBJECTIVE))

# ## Run the optimization loop
#
# We can now run the optimization loop

num_steps = 20
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
result = bo.optimize(num_steps, initial_data, models, acquisition_rule=rule)

# To conclude, we visualize the multi-objective bo queried data in the design space

# +
datasets = result.try_get_final_datasets()
data_query_points = datasets[OBJECTIVE].query_points.numpy()
data_observations = datasets[OBJECTIVE].observations.numpy()

_, ax = plot_function_2d(
    vlmop2, mins, maxs, grid_density=100, contour=True, title=["Obj 1", "Obj 2"]
)
plot_bo_points(data_query_points, ax=ax[0, 0], num_init=num_initial_points)
plot_bo_points(data_query_points, ax=ax[0, 1], num_init=num_initial_points)
plt.show()
# -

# ... and visulize in the objective space, purple dots denotes the nondominated points.

plot_bo_points_in_obj_space(data_observations, num_init=num_initial_points)
plt.show()

# We can also visualize how the performance metric get envolved with respect to the bo iterations.  
# First, we need to define a metric function by ourself. The performance metric function for multi-objective function can be fleaxibly choosen. Here, we make use of the log hypervolume difference.

# The log hypervolume difference is defined as the difference between the hypervolume of the true Pareto front and the hypervolume of the approximate Pareto front based on the observed data.
# $$
# log_{10}\ \text{HV}_{diff} = log_{10}(\text{HV}_{\text{true}} - \text{HV}_{\text{bo obtained}})
# $$
#

# First we need to calculate the $\text{HV}_{\text{ideal}}$, this is approximated by calculating the hypervolume based on ideal pareto front:

ideal_pf = VLMOP2().gen_pareto_optimal_points(100).numpy().astype(data_observations.dtype)  # gen 100 pf points
ref_point = get_reference_point(data_observations).numpy()
idea_hv = Pareto(ideal_pf).hypervolume_indicator(ref_point)


# Then define the metric function:

def log_hv(observations):
    obs_hv = Pareto(observations).hypervolume_indicator(ref_point)
    return math.log(idea_hv - obs_hv)


fig, ax = plot_mo_history(data_observations, log_hv, num_init = num_initial_points)
ax.set_xlabel('Iterations')
ax.set_ylabel('log HV difference')
plt.show()

# ## LICENSE
#
# [Apache License 2.0](https://github.com/secondmind-labs/trieste/blob/develop/LICENSE)
