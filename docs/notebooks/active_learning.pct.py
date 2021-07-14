# ### Active Learning

# +
# %matplotlib inline
import numpy as np
import tensorflow as tf
import pandas as pd

np.random.seed(1793)
tf.random.set_seed(1793)
# -

# ### The Problem
#
# active learning is bla bla, in this notebook we will bla bla using bla bla


# +
from trieste.utils.objectives import branin
from util.plotting_plotly import plot_function_plotly
from trieste.space import Box

def log_branin(x):
    return tf.math.log(branin(x))

search_space = Box([0, 0], [1, 1])

fig = plot_function_plotly(log_branin, search_space.lower, search_space.upper, grid_density=20)
fig.update_layout(height=400, width=400)
fig.show()

# +
import trieste

observer = trieste.utils.objectives.mk_observer(log_branin)

num_initial_points = 2
initial_query_points = search_space.sample_halton(num_initial_points)
initial_data = observer(initial_query_points)


# +
import gpflow
from trieste.acquisition.optimizer import generate_continuous_optimizer
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.acquisition.function import PredictiveVariance

def build_model(data):
    variance = tf.math.reduce_variance(data.observations)
    kernel = gpflow.kernels.RBF(variance=variance, lengthscales=[0.2, 0.2])
    gpr = gpflow.models.GPR(data.astuple(), kernel, noise_variance=1e-5)
    gpflow.set_trainable(gpr.likelihood, False)

    return {
            "model": gpr,
            "optimizer": gpflow.optimizers.Scipy(),
            "optimizer_args": {
                "minimize_args": {"options": dict(maxiter=100)},
            },
    }

model = build_model(initial_data)
# -

# ### Active Learning using Predictive Variance

acq = PredictiveVariance()
rule = EfficientGlobalOptimization(builder=acq, optimizer=generate_continuous_optimizer(sigmoid=False))
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer,search_space)

# +
import copy
dataset = copy.deepcopy(initial_data)
model_evolution= []
model_temp = model
bo_iter = 5

#optimize bo once at iteration for capturing model 
for i in range(bo_iter):
    result = bo.optimize(1, dataset, model_temp, rule)
    dataset = result.try_get_final_dataset()
    model_temp = copy.deepcopy(result.try_get_final_model())
    model_evolution.append(model_temp)
# -

query_points = dataset.query_points.numpy()
observations = dataset.observations.numpy()

# +
from util.plotting import plot_bo_points, plot_function_2d

for i in range(bo_iter):
    
    def pred_var(x):
        _, var = model_evolution[i].model.predict_f(x)
        return var

    _, ax = plot_function_2d(
        pred_var, search_space.lower, search_space.upper, grid_density=20, contour=True, 
        colorbar=True,     
        figsize=(10, 6),
        title=["Variance contour with queried points at iter:"+str(i+1)],
        xlabel="$X_1$",
        ylabel="$X_2$",
    )
    plot_bo_points(query_points[:num_initial_points+i+1], ax[0, 0], num_initial_points)

# -

# ### Batch Active Learning using Predictive Variance

# +
num_query = 3
acq = PredictiveVariance()
rule = EfficientGlobalOptimization(num_query_points=num_query, builder=acq, optimizer=generate_continuous_optimizer(sigmoid=False))
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer,search_space)

import copy
dataset = copy.deepcopy(initial_data)
model_evolution= []
model_temp = model
bo_iter = 5

#optimize bo once at iteration for capturing model 
for i in range(bo_iter):
    result = bo.optimize(1, dataset, model_temp, rule)
    dataset = result.try_get_final_dataset()
    model_temp = copy.deepcopy(result.try_get_final_model())
    model_evolution.append(model_temp)
    
query_points = dataset.query_points.numpy()
observations = dataset.observations.numpy()

from util.plotting import plot_bo_points, plot_function_2d

for i in range(bo_iter):
    
    def pred_var(x):
        _, var = model_evolution[i].model.predict_f(x)
        return var

    _, ax = plot_function_2d(
        pred_var, search_space.lower-0.1, search_space.upper+0.1, grid_density=20, contour=True, 
        colorbar=True,     
        figsize=(10, 6),
        title=["Variance contour with queried points at iter:"+str(i+1)],
        xlabel="$X_1$",
        ylabel="$X_2$",
    )
    plot_bo_points(query_points[:num_initial_points+(i+1)*num_query], ax[0, 0], num_initial_points)

# -


