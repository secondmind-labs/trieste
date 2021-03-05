# ... and in the objective space
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
from trieste.acquisition.multiobjective.analytic import Expected_Hypervolume_Improvement

from util.plotting import plot_function_2d, plot_bo_points

np.random.seed(1793)
tf.random.set_seed(1793)

from util.plotting import plot_bo_points_in_obj_space


# ## Modelling the two functions
#
# We'll model the different objective functions with their own Gaussian process regression models.

def create_bo_model(data, input_dim=2, l=1.0):
    variance = tf.math.reduce_variance(data.observations)
    lengthscale = l * np.ones(input_dim, dtype=gpflow.default_float())
    kernel = gpflow.kernels.Matern52(variance=variance, lengthscales=lengthscale)
    gpr = gpflow.models.GPR(data.astuple(), kernel, noise_variance=1e-3)
    return create_model({
        "model": gpr,
        "optimizer": gpflow.optimizers.Scipy(),
        "optimizer_args": {
            "minimize_args": {"options": dict(maxiter=100)},
        },
    })


# # Advanced: Problem with 3 Objective Function

# Now we demonstrate an optimization for DTLZ2 function with 3 objectives in 4 dimension:

# +
from tensorflow import sin, cos
from math import pi


def dtlz2(x: TensorType, M: int = 3) -> TensorType:
    """
    DTLZ test problem 2.
    """

    def g(xm):
        z = xm - 0.5
        return tf.reduce_sum(z ** 2, axis=1, keepdims=True)

    def problem2(x, M):
        f = None

        for i in range(M):
            y = (1 + g(x[:, M - 1:]))
            for j in range(M - 1 - i):
                y *= cos((pi * x[:, j, tf.newaxis]) / 2)
            if i > 0:
                y *= sin((pi * x[:, M - 1 - i, tf.newaxis]) / 2)
            f = y if f is None else tf.concat([f, y], 1)
        return f

    return problem2(x, M)


def observer(query_points):
    y = dtlz2(query_points)
    return {OBJECTIVE: trieste.data.Dataset(query_points, y)}


# -

# Now we can follow similar setup to optimize this problem, it will take a while waiting for the finish of the optimizer

# +
input_dim = 12
mins = [0] * input_dim
maxs = [1] * input_dim
lower_bound = tf.cast(mins, gpflow.default_float())
upper_bound = tf.cast(maxs, gpflow.default_float())
search_space = trieste.space.Box(lower_bound, upper_bound)

num_objective = 3

num_initial_points = 26
initial_query_points = search_space.sample(num_initial_points)
initial_data = observer(initial_query_points)

objective_models = [(create_bo_model(Dataset(initial_data[OBJECTIVE].query_points,
                                             tf.gather(initial_data[OBJECTIVE].observations, [i], axis=1))
                                     , input_dim, 1.0), 1) for i in range(num_objective)]

models = {OBJECTIVE: ModelStack(*objective_models)}

hvei = Expected_Hypervolume_Improvement().using(OBJECTIVE)
rule = trieste.acquisition.rule.EfficientGlobalOptimization(builder=hvei)

num_steps = 50
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
result = bo.optimize(num_steps, initial_data, models, acquisition_rule=rule)
# -

plot_bo_points_in_obj_space(result.try_get_final_datasets()[OBJECTIVE], num_init=num_initial_points)
plt.show()
