import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import trieste
from trieste.data import Dataset, split_dataset_by_fidelity, check_and_extract_fidelity_query_points
from trieste.models.gpflow.models import MultifidelityAutoregressive
from trieste.models.gpflow.builders import build_multifidelity_autoregressive_models
from trieste.objectives import Linear2Fidelity

np.random.seed(17943)
tf.random.set_seed(17943)

from trieste.types import TensorType
from trieste.objectives.utils import mk_observer

linear_two_fidelity = Linear2Fidelity.objective
n_fidelities = Linear2Fidelity.num_fidelities
input_search_space = Linear2Fidelity.input_search_space
search_space = Linear2Fidelity.search_space


def noisy_linear_2_fidelity(x: TensorType) -> TensorType:

    _, fidelities = check_and_extract_fidelity_query_points(x)
    y = linear_two_fidelity(x)
    not_lowest_fidelity = fidelities > 0
    noise = tf.random.normal(y.shape, stddev=1e-1, dtype=y.dtype)
    y = tf.where(not_lowest_fidelity, y + noise, y)
    return y


observer = mk_observer(noisy_linear_2_fidelity)

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
noise_free_observer = mk_observer(linear_two_fidelity)
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
