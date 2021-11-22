# %% [markdown]
# # Active Learning for Gaussian Process Classification Model

# %%
from dataclasses import astuple

import gpflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from gpflow.utilities import print_summary, set_trainable

import trieste
from trieste.acquisition.function import BayesianActiveLearningByDisagreement, PredictiveVariance
from trieste.acquisition.rule import OBJECTIVE
from trieste.models import create_model
from trieste.models.gpflow.models import GaussianProcessRegression, VariationalGaussianProcess
from trieste.objectives.utils import mk_observer
from trieste.utils import map_values

np.random.seed(1234)
tf.random.set_seed(1234)

search_space = trieste.space.Box([-1, -1], [1, 1])
input_dim = 2


def circle(x):
    return tf.cast((tf.reduce_sum(tf.square(x), axis=1, keepdims=True) - 0.5) > 0, tf.float64)


# %% [markdown]
# ## Describe the problem

# %%
from trieste.models.gpflow import GPflowModelConfig, VariationalGaussianProcess


def create_bo_model(data):
    kernel = gpflow.kernels.SquaredExponential()
    m = gpflow.models.VGP(astuple(data), likelihood=gpflow.likelihoods.Bernoulli(), kernel=kernel)
    return trieste.models.create_model(
        GPflowModelConfig(
            **{
                "model": m,
                "optimizer": gpflow.optimizers.Scipy(),
                "optimizer_args": {
                    "minimize_args": {"options": dict(maxiter=100.0)},
                },
            }
        )
    )


"""class scipyminimizer():
    def optimize(self, model, data):
        opt = gpflow.optimizers.Scipy()
        opt.minimize(model.training_loss, variables=model.trainable_variables)
        
def create_bo_model(data):
    kernel = gpflow.kernels.SquaredExponential()
    m = gpflow.models.VGP(astuple(data), likelihood=gpflow.likelihoods.Bernoulli(), kernel=kernel)
    return VariationalGaussianProcess(m,scipyminimizer())
"""

# %%
density = 100
xx = np.linspace(search_space.lower[0], search_space.upper[0], density)

grid_xx = np.meshgrid(*[xx] * input_dim)
xx = np.vstack([g.ravel() for g in grid_xx]).T
yy = circle(xx).numpy()

plt.figure()
plt.contour(*grid_xx, np.reshape(yy, [density] * input_dim), levels=[0.5])
idx = np.squeeze(yy).astype(bool)
plt.scatter(xx[idx][:, 0], xx[idx][:, 1], label="1")
plt.scatter(xx[np.logical_not(idx)][:, 0], xx[np.logical_not(idx)][:, 1], label="0")
plt.legend()
plt.show()

numSamples = 10
X = search_space.sample_halton(numSamples)


observer = mk_observer(circle, OBJECTIVE)
datasets = observer(X)

# %%
model = create_bo_model(datasets[OBJECTIVE])

model.update(datasets[OBJECTIVE])
model.optimize(datasets[OBJECTIVE])

mean, variance = model.predict(xx)

plt.figure()
plt.contourf(*grid_xx, np.reshape(mean, [density] * input_dim))
plt.plot(
    datasets[OBJECTIVE].query_points[:, 0],
    datasets[OBJECTIVE].query_points[:, 1],
    "ko",
    markersize=10,
)
plt.title("Mean")
plt.colorbar()
plt.show()

plt.figure()
plt.contourf(*grid_xx, np.reshape(variance, [density] * input_dim))
plt.colorbar()
plt.plot(
    datasets[OBJECTIVE].query_points[:, 0],
    datasets[OBJECTIVE].query_points[:, 1],
    "ko",
    markersize=10,
)
plt.title("Variance")
plt.show()


# %%
initial_models = trieste.utils.map_values(create_bo_model, datasets)

# %%
acq = BayesianActiveLearningByDisagreement()
# %% BO
numIter = 50
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
rule = trieste.acquisition.rule.EfficientGlobalOptimization(acq.using(OBJECTIVE))

results = bo.optimize(numIter, datasets, initial_models, rule, track_state=False)

# %%
final_dataset = results.try_get_final_datasets()[OBJECTIVE]
final_model = results.try_get_final_models()[OBJECTIVE]

xmax = final_dataset.query_points[-numIter:, :]

# %% Plot BO results
mean, variance = final_model.predict(xx)

plt.figure(figsize=(10, 8))
plt.contourf(*grid_xx, np.reshape(mean, [density] * input_dim))
plt.colorbar()
plt.plot(
    final_dataset.query_points[:-numIter, 0],
    final_dataset.query_points[:-numIter, 1],
    "ko",
    markersize=10,
)
plt.plot(
    final_dataset.query_points[-numIter:, 0], final_dataset.query_points[-numIter:, 1], "rx", mew=10
)
plt.contour(*grid_xx, np.reshape(yy, [density] * input_dim), levels=[0.5])
plt.title("Updated Mean")
plt.show()

# %%
