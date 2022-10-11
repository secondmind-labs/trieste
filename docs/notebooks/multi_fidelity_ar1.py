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
import tensorflow_probability as tfp
import numpy as np
from typing import TypeVar
from trieste.data import Dataset
from trieste.models.gpflow.models import AR1
from trieste.types import TensorType
from trieste.models.gpflow.builders import build_ar1_models
import matplotlib.pyplot as plt

OBJECTIVE = "OBJECTIVE"

ProbabilisticModelType = TypeVar(
    "ProbabilisticModelType", bound="ProbabilisticModel", contravariant=True
)
# from tensorflow.python.ops.numpy_ops import np_config
tf.experimental.numpy.experimental_enable_numpy_behavior()  # NOTE: This depends on tf 2.5 and trieste currently depends on 2.4
# np_config.enable_numpy_behavior()

# + pycharm={"name": "#%%\n"}
# Should live in its own file, i.e. multifidelity dataset and associated utilities


# + pycharm={"name": "#%%\n"}


# + pycharm={"name": "#%%\n"}

# Replace this with your own observer
def my_simulator(x_input, fidelity, add_noise):
    # this is a dummy objective
    f = 0.5 * ((6.0 * x_input - 2.0) ** 2) * tf.math.sin(12.0 * x_input - 4.0) + 10.0 * (
        x_input - 1.0
    )
    f = f + fidelity * (f - 20.0 * (x_input - 1.0))
    if add_noise:
        noise = tf.random.normal(f.shape, stddev=1e-1, dtype=f.dtype)
    else:
        noise = 0
    f = tf.where(fidelity > 0, f + noise, f)
    return f


def observer(x, num_fidelities, add_noise=True):
    # last dimension is the fidelity value
    x_input = x[..., :-1]
    x_fidelity = x[..., -1:]

    # note: this assumes that my_simulator broadcasts, i.e. accept matrix inputs.
    # If not you need to replace this by a for loop over all rows of "input"
    observations = my_simulator(x_input, x_fidelity, add_noise)
    return Dataset(query_points=x, observations=observations)


input_dim = 1
lb = np.zeros(input_dim)
ub = np.ones(input_dim)
n_fidelities = 4

input_search_space = trieste.space.Box(lb, ub)
fidelity_search_space = trieste.space.DiscreteSearchSpace(
    np.array([np.arange(n_fidelities, dtype=float)]).reshape(-1, 1)
)
search_space = trieste.space.TaggedProductSearchSpace(
    [input_search_space, fidelity_search_space], ["input", "fidelity"]
)
n_samples_per_fidelity = [
    2 ** ((n_fidelities - fidelity) + 1) + 3 for fidelity in range(n_fidelities)
]

xs = [tf.linspace(0, 1, samples)[:, None] for samples in n_samples_per_fidelity]
initial_samples_list = [tf.concat([x, tf.ones_like(x) * i], 1) for i, x in enumerate(xs)]
initial_sample = tf.concat(initial_samples_list, 0)
initial_data = observer(initial_sample, num_fidelities=n_fidelities)

# points, _, indices = filter_by_fidelity(initial_data.query_points)
# data = [
#     Dataset(points[fidelity], tf.gather(initial_data.observations, indices[fidelity]))
#     for fidelity in range(n_fidelities)
# ]

model = AR1(
    build_ar1_models(initial_data, n_fidelities, input_search_space)
)

model.update(initial_data)
model.optimize(initial_data)


X = tf.linspace(0, 1, 200)[:, None]
X_list = [tf.concat([X, tf.ones_like(X) * i], 1) for i in range(n_fidelities)]
predictions = [model.predict(x) for x in X_list]
fig, ax = plt.subplots(1, 1, figsize=(10, 7))
for fidelity, prediction in enumerate(predictions):
    mean, var = prediction
    ax.plot(X, mean, label=f"Predicted fidelity {fidelity}")
    ax.plot(X, mean + 1.96 * tf.math.sqrt(var), alpha=0.2)
    ax.plot(X, mean - 1.96 * tf.math.sqrt(var), alpha=0.2)
    ax.plot(
        X,
        observer(X_list[fidelity], num_fidelities=n_fidelities, add_noise=False).observations,
        label=f"True fidelity {fidelity}",
    )
    ax.scatter(
        data[fidelity].query_points,
        data[fidelity].observations,
        label=f"fidelity {fidelity} data",
    )
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
print(f"Optimised rho for fidelity 2 as {model.rho[1].numpy()}")
print(f"Optimised rho for fidelity 3 as {model.rho[2].numpy()}")
# print(f"Optimised rho for fidelity 4 as {model.rho[3].numpy()}")
plt.show()

# + pycharm={"name": "#%%\n"}

# X_fids = tf.concat([X, tf.ones_like(X) * 3], 1)
# samples_0 = tf.squeeze(model.sample(X_fids, 1))
# samples_1 = tf.squeeze(model.sample(X_fids, 1))
# fig, ax = plt.subplots(1, 1, figsize=(10, 7))
# ax.scatter(X, samples_0, marker="x")
# ax.scatter(X, samples_1, marker="x")
# ax.plot(
#     X,
#     observer(X_fids, num_fidelities=n_fidelities, add_noise=False).observations,
#     label=f"True fidelity {fidelity}",
# )
# plt.show()
