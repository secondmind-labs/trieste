import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import trieste
from trieste.data import Dataset, split_dataset_by_fidelity
from trieste.models.gpflow.models import MultifidelityAutoregressive
from trieste.models.gpflow.builders import build_multifidelity_autoregressive_models

np.random.seed(17943)
tf.random.set_seed(17943)

from trieste.types import TensorType


def linear_multifidelity(x: TensorType):

    x_input = x[..., :-1]
    x_fidelity = x[..., -1:]

    f = 0.5 * ((6.0 * x_input - 2.0) ** 2) * tf.math.sin(12.0 * x_input - 4.0) + 10.0 * (
        x_input - 1.0
    )
    f = f + x_fidelity * (f - 20.0 * (x_input - 1.0))
    noise = tf.random.normal(f.shape, stddev=1e-1, dtype=f.dtype)
    f = f + noise

    return f


def noise_free_linear_multifidelity(x: TensorType):

    x_input = x[..., :-1]
    x_fidelity = x[..., -1:]

    f = 0.5 * ((6.0 * x_input - 2.0) ** 2) * tf.math.sin(12.0 * x_input - 4.0) + 10.0 * (
        x_input - 1.0
    )
    f = f + x_fidelity * (f - 20.0 * (x_input - 1.0))

    return f


from trieste.objectives.utils import mk_observer

observer = mk_observer(linear_multifidelity)
input_dim = 1
lb = np.zeros(input_dim)
ub = np.ones(input_dim)
n_fidelities = 2

input_search_space = trieste.space.Box(lb, ub)
fidelity_search_space = trieste.space.DiscreteSearchSpace(
    np.array([np.arange(n_fidelities, dtype=float)]).reshape(-1, 1)
)
search_space = trieste.space.TaggedProductSearchSpace(
    [input_search_space, fidelity_search_space], ["input", "fidelity"]
)


n_samples_per_fidelity = [11, 6]
low_cost = 1.0
high_cost = 1.0

xs = [tf.random.uniform([n_samples_per_fidelity[0], 1], 0, 1, dtype=tf.float64)]
hf_samples = tf.Variable(
    np.random.choice(xs[0][:, 0], size=n_samples_per_fidelity[1], replace=False)
)[:, None]
xs.append(hf_samples)

initial_samples_list = [tf.concat([x, tf.ones_like(x) * i], 1) for i, x in enumerate(xs)]
initial_sample = tf.concat(initial_samples_list, axis=0)
initial_data = observer(initial_sample)

model = MultifidelityAutoregressive(
    build_multifidelity_autoregressive_models(initial_data, n_fidelities, input_search_space)
)

model.update(initial_data)
model.optimize(initial_data)

data = split_dataset_by_fidelity(initial_data, n_fidelities)
noise_free_observer = mk_observer(noise_free_linear_multifidelity)
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
        noise_free_observer(X_list[fidelity]).observations,
        label=f"True fidelity {fidelity}",
    )
    ax.scatter(
        data[fidelity].query_points,
        data[fidelity].observations,
        label=f"fidelity {fidelity} data",
    )
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.show()

# Set up bayes opt loop
from trieste.acquisition.combination import Product
from trieste.acquisition.function.entropy import MUMBO, CostWeighting
from trieste.space import SearchSpace
from trieste.acquisition.optimizer import generate_continuous_optimizer

bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
acq_builder = Product(
    MUMBO(search_space).using("OBJECTIVE"), CostWeighting(low_cost, high_cost).using("OBJECTIVE")
)
optimizer = generate_continuous_optimizer(num_initial_samples=10_000, num_optimization_runs=10)
rule = trieste.acquisition.rule.EfficientGlobalOptimization(builder=acq_builder)

num_steps = 1
result = bo.optimize(num_steps, initial_data, model, acquisition_rule=rule)

dataset = result.try_get_final_dataset()

data = split_dataset_by_fidelity(dataset, n_fidelities)
X = tf.linspace(0, 1, 200)[:, None]
X_list = [tf.concat([X, tf.ones_like(X) * i], 1) for i in range(n_fidelities)]
predictions = [model.predict(x) for x in X_list]
fig, ax = plt.subplots(1, 1, figsize=(7, 4))
for fidelity, prediction in enumerate(predictions):
    mean, var = prediction
    ax.plot(X, mean, label=f"Predicted fidelity {fidelity}")
    ax.plot(X, mean + 1.96 * tf.math.sqrt(var), alpha=0.2)
    ax.plot(X, mean - 1.96 * tf.math.sqrt(var), alpha=0.2)
    ax.plot(
        X,
        noise_free_observer(X_list[fidelity]).observations,
        label=f"True fidelity {fidelity}",
    )
    ax.scatter(
        data[fidelity].query_points,
        data[fidelity].observations,
        label=f"fidelity {fidelity} data",
    )
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))


# plot final acquisition function
fig, ax = plt.subplots(1, 1, figsize=(7, 4))
X = tf.linspace(0, 1, 1000)[:, None]
X_low = tf.expand_dims(tf.concat([X, tf.zeros_like(X)], 1), 1)
X_high = tf.expand_dims(tf.concat([X, tf.ones_like(X)], 1), 1)
ax.plot(X, rule._acquisition_function(X_low), label="acq_low")
ax.plot(X, rule._acquisition_function(X_high), label="acq high")
ax.legend()
