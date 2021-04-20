# -*- coding: utf-8 -*-
# # Multi-objective optimization with Expected HyperVolume Improvement Approach

import gpflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from util.plotting import plot_bo_points, plot_function_2d

# +
import trieste
from trieste.acquisition.function import ExpectedHypervolumeImprovement
from trieste.acquisition.rule import OBJECTIVE
from triestehttp://localhost:8888/notebooks/multiobjective_ehvi.pct.py#.data import Dataset
from trieste.models import create_model
from trieste.models.model_interfaces import ModelStack
from trieste.space import Box
from trieste.utils.mo_objectives import VLMOP2

np.random.seed(1793)
tf.random.set_seed(1793)


# -

# ## Describe the problem
#
# In this tutorial, we provide multi-objective optimization example using the expected hypervolume improvement acquisition function from [1]. The synthetic function: VLMOP2 is a functions with 2 outcomes. We'll start by defining the problem parameters.


vlmop2 = VLMOP2().prepare_benchmark()
observer = trieste.utils.objectives.mk_observer(vlmop2, OBJECTIVE)

mins = [-2, -2]
maxs = [2, 2]
search_space = Box(mins, maxs)
num_objective = 2

# Let's randomly sample some initial data from the observer ...

num_initial_points = 10
initial_query_points = search_space.sample(num_initial_points)
initial_data = observer(initial_query_points)

# ... and visualise those points in the design space.

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
# We'll model the different objective functions with their own Gaussian process regression models.


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

models = {OBJECTIVE: ModelStack(*objective_models)}

# ## Define the acquisition process
#
# Here we utilize the `HVExpectedImprovement` acquisition function proposed in
# Yang 2019 [1]:

from trieste.acquisition.rule import EfficientGlobalOptimization

hvei = ExpectedHypervolumeImprovement()
rule: EfficientGlobalOptimization[Box] = EfficientGlobalOptimization(builder=hvei.using(OBJECTIVE))

# ## Run the optimization loop
#
# We can now run the optimization loop

num_steps = 20
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
result = bo.optimize(num_steps, initial_data, models, acquisition_rule=rule)

# To conclude, we visualize the queried data in the design space

# +
datasets = result.try_get_final_datasets()
data_query_points = datasets[OBJECTIVE].query_points

_, ax = plot_function_2d(
    vlmop2, mins, maxs, grid_density=100, contour=True, title=["Obj 1", "Obj 2"]
)
plot_bo_points(data_query_points, ax=ax[0, 0], num_init=num_initial_points)
plot_bo_points(data_query_points, ax=ax[0, 1], num_init=num_initial_points)
plt.show()
# -

# ... and visulize in the objective space, orange dots denotes the nondominated points.

plot_bo_points_in_obj_space(datasets[OBJECTIVE].observations, num_init=num_initial_points)
plt.show()

# [1] Yang, K., Emmerich, M., Deutz, A., & Bäck, T. (2019). Efficient computation of expected hypervolume improvement using box decomposition algorithms. Journal of Global Optimization, 75(1), 3-34.

# ## LICENSE
#
# [Apache License 2.0](https://github.com/secondmind-labs/trieste/blob/develop/LICENSE)
