# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3.7.13 64-bit ('multifidelity')
#     language: python
#     name: python3
# ---

# + pycharm={"name": "#%%\n"}
from __future__ import annotations

import trieste
import tensorflow as tf
import numpy as np
from typing import TypeVar
from trieste.data import Dataset, MultifidelityDataset
from trieste.models.gpflow.models import AR1
from trieste.types import TensorType
from trieste.models.gpflow.builders import build_gpr
from trieste.models.gpflow import GaussianProcessRegression
import matplotlib.pyplot as plt
OBJECTIVE = "OBJECTIVE"

ProbabilisticModelType = TypeVar(
    "ProbabilisticModelType", bound="ProbabilisticModel", contravariant=True
)
#from tensorflow.python.ops.numpy_ops import np_config
tf.experimental.numpy.experimental_enable_numpy_behavior()  # NOTE: This depends on tf 2.5 and trieste currently depends on 2.4
#np_config.enable_numpy_behavior()

# + pycharm={"name": "#%%\n"}
# Should live in its own file, i.e. multifidelity dataset and associated utilities


# + pycharm={"name": "#%%\n"}


# + pycharm={"name": "#%%\n"}

def filter_by_fidelity(query_points: TensorType):

    input_points = query_points[:, :-1]  # [..., D+1]
    fidelities = query_points[:, -1]  # [..., 1]
    max_fid = int(tf.reduce_max(fidelities))
    masks = list()
    indices = list()
    points = list()
    for fidelity in range(max_fid+1):
        fid_mask = (fidelities == fidelity)
        fid_ind = tf.where(fid_mask)[:, 0]
        fid_points = tf.gather(input_points, fid_ind, axis=0)
        masks.append(fid_mask)
        indices.append(fid_ind)
        points.append(fid_points)
    return points, masks, indices

# Replace this with your own observer
def my_simulator(x_input, fidelity):
    # this is a dummy objective
    f = 0.5 * ((6.0*x_input-2.0)**2)*tf.math.sin(12.0*x_input - 4.0) + 10.0*(x_input -1.0)
    f = f + fidelity * (f - 20.0*(x_input -1.0))
    noise = tf.random.normal(f.shape, stddev=1e-1, dtype=f.dtype)
    f = tf.where(fidelity > 0, f + noise, f)
    return f


def observer(x, num_fidelities):
    # last dimension is the fidelity value
    x_input = x[..., :-1]
    x_fidelity = x[...,-1:]

    # note: this assumes that my_simulator broadcasts, i.e. accept matrix inputs.
    # If not you need to replace this by a for loop over all rows of "input"
    observations = my_simulator(x_input, x_fidelity)
    return MultifidelityDataset(num_fidelities=num_fidelities, query_points=x, observations=observations)


input_dim = 1
lb = np.zeros(input_dim)
ub = np.ones(input_dim)
n_fidelities = 4

input_search_space = trieste.space.Box(lb, ub)
fidelity_search_space = trieste.space.DiscreteSearchSpace(np.array([np.arange(n_fidelities, dtype=float)]).reshape(-1, 1))
search_space = trieste.space.TaggedProductSearchSpace([input_search_space, fidelity_search_space],
                                                      ["input", "fidelity"])
n_samples_per_fidelity = [2**((n_fidelities - fidelity) + 1) + 3 for fidelity in range(n_fidelities)]

xs = [tf.linspace(0, 1, samples)[:, None] for samples in n_samples_per_fidelity]
initial_samples_list = [tf.concat([x, tf.ones_like(x) * i], 1) for i, x in enumerate(xs)]
initial_sample = tf.concat(initial_samples_list,0)
initial_data = observer(initial_sample, num_fidelities=n_fidelities)

points, masks, indices = filter_by_fidelity(initial_data.query_points)
data = [Dataset(points[fidelity], tf.gather(initial_data.observations, indices[fidelity])) for fidelity in range(n_fidelities)]

gprs = [GaussianProcessRegression(build_gpr(data[fidelity], input_search_space,  likelihood_variance = 1e-6, kernel_priors=False)) for fidelity in range(n_fidelities)]

model = AR1(
    lowest_fidelity_signal_model = gprs[0],
    fidelity_residual_models = gprs[1:],

)

model.update(initial_data)
model.optimize(initial_data)


X = tf.linspace(0,1,200)[:, None]
X_list = [tf.concat([X, tf.ones_like(X) * i], 1) for i in range(n_fidelities)]
predictions = [model.predict(x) for x in X_list]
fig, ax = plt.subplots(1,1, figsize=(10, 7))
for fidelity, prediction in enumerate(predictions):
    mean, var = prediction
    ax.plot(X,mean, label=f"Predicted fidelity {fidelity}")
    ax.plot(X,mean+1.96*tf.math.sqrt(var), alpha=0.2)
    ax.plot(X,mean-1.96*tf.math.sqrt(var), alpha=0.2)
    ax.plot(X,observer(X_list[fidelity], num_fidelities=n_fidelities).observations, label=f"True fidelity {fidelity}")
    ax.scatter(points[fidelity], tf.gather(initial_data.observations, indices[fidelity]), label=f"fidelity {fidelity} data")
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
print(f"Optimised rho for fidelity 2 as {model.rho[1].numpy()}")
print(f"Optimised rho for fidelity 3 as {model.rho[2].numpy()}")
#print(f"Optimised rho for fidelity 4 as {model.rho[3].numpy()}")
plt.show()

# + pycharm={"name": "#%%\n"}

