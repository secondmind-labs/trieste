# -*- coding: utf-8 -*-
# # Multi-objective optimization: an Expected HyperVolume Improvement Approach

# +
import trieste
import gpflow
import numpy as np
import tensorflow as tf
from trieste.type import TensorType
from trieste.data import Dataset
from trieste.acquisition.rule import OBJECTIVE
from trieste.models.model_interfaces import ModelStack
from trieste.models import create_model
import matplotlib.pyplot as plt
from trieste.acquisition.multiobjective.qEHVI import BatchMonteCarloHypervolumeExpectedImprovement

from util.plotting import plot_function_2d, plot_bo_points

np.random.seed(1793)
tf.random.set_seed(1793)


# -

# ## The problem
#
# In this tutorial, we replicate one of the numerical examples in [GPflowOpt](https://github.com/GPflow/GPflowOpt/blob/master/doc/source/notebooks/multiobjective.ipynb) using acquisition function from Couckuyt, 2014 [1], which is a multi-objective optimization problem with 2 objective functions. We'll start by defining the problem parameters.

def vlmop2(x: TensorType) -> TensorType:
    transl = 1 / np.sqrt(2)
    part1 = (x[:, 0] - transl) ** 2 + (x[:, 1] - transl) ** 2
    part2 = (x[:, 0] + transl) ** 2 + (x[:, 1] + transl) ** 2
    y1 = 1 - tf.exp(-1 * part1)
    y2 = 1 - tf.exp(-1 * part2)
    return tf.stack([y1, y2], axis=1)


mins = [-2, -2]
maxs = [2, 2]
lower_bound = tf.cast(mins, gpflow.default_float())
upper_bound = tf.cast(maxs, gpflow.default_float())
search_space = trieste.space.Box(lower_bound, upper_bound)

# We'll make an observer that outputs different objective function values, labelling each as shown.

num_objective = 2


def observer(query_points):
    y = vlmop2(query_points)
    return {OBJECTIVE: trieste.data.Dataset(query_points, y)}


# Let's randomly sample some initial data from the observer ...

num_initial_points = 10
initial_query_points = search_space.sample(num_initial_points)
initial_data = observer(initial_query_points)

# ... and visualise those points in the design space.

_, ax = plot_function_2d(vlmop2, mins, maxs, grid_density=100, contour=True, title=['Obj 1', 'Obj 2'])
plot_bo_points(initial_query_points, ax=ax[0, 0], num_init=num_initial_points)
plot_bo_points(initial_query_points, ax=ax[0, 1], num_init=num_initial_points)
plt.show()

# ... and in the objective space

from util.plotting import plot_bo_points_in_obj_space

plot_bo_points_in_obj_space(initial_data[OBJECTIVE])
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
    return create_model({
        "model": gpr,
        "optimizer": gpflow.optimizers.Scipy(),
        "optimizer_args": {
            "minimize_args": {"options": dict(maxiter=100)},
        },
    })


objective_models = [(create_bo_model(Dataset(initial_data[OBJECTIVE].query_points,
                                             tf.gather(initial_data[OBJECTIVE].observations, [i], axis=1))), 1) \
                    for i in range(num_objective)]

models = {OBJECTIVE: ModelStack(*objective_models)}

# ## Define the acquisition process
#
# Here we utilize the `HVExpectedImprovement` acquisition function proposed in
# Yang 2019 [1]:

qehvi = BatchMonteCarloHypervolumeExpectedImprovement().using(OBJECTIVE)
rule = trieste.acquisition.rule.BatchAcquisitionRule(num_query_points=3, builder=qehvi)

# ## Run the optimization loop
#
# We can now run the optimization loop

num_steps = 10
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
result = bo.optimize(num_steps, initial_data, models, acquisition_rule=rule)

# To conclude, we visualize the queried data in the design space

# +
datasets = result.try_get_final_datasets()
data_query_points = datasets[OBJECTIVE].query_points

_, ax = plot_function_2d(vlmop2, mins, maxs, grid_density=100, contour=True, title=['Obj 1', 'Obj 2'])
plot_bo_points(data_query_points, ax=ax[0, 0], num_init=num_initial_points)
plot_bo_points(data_query_points, ax=ax[0, 1], num_init=num_initial_points)
plt.show()
# -

# ... and visulize in the objective space, orange dots denotes the nondominated points.

plot_bo_points_in_obj_space(datasets[OBJECTIVE], num_init=num_initial_points)
plt.show()

# [1] Yang, K., Emmerich, M., Deutz, A., & Bäck, T. (2019). Efficient computation of expected hypervolume improvement using box decomposition algorithms. Journal of Global Optimization, 75(1), 3-34.

# ## LICENSE
#
# [Apache License 2.0](https://github.com/secondmind-labs/trieste/blob/develop/LICENSE)
