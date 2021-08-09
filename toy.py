# Mirrors the EI tutorial; playing around
import numpy as np
import tensorflow as tf
from trieste.objectives import branin
from notebooks.util.plotting_plotly import (
    plot_function_plotly,
    add_bo_points_plotly,
    plot_gp_plotly,
)
from notebooks.util.plotting import plot_bo_points, plot_function_2d
import trieste
from trieste.space import Box
import gpflow
import matplotlib.pyplot as plt
from notebooks.util.plotting import plot_regret

np.random.seed(1793)
tf.random.set_seed(1793)

search_space = Box([0, 0], [1, 1])

fig = plot_function_plotly(branin, search_space.lower, search_space.upper, grid_density=20)
fig.update_layout(height=400, width=400)
fig.show()

observer = trieste.objectives.utils.mk_observer(branin)

num_initial_points = 50
initial_query_points = search_space.sample(num_initial_points)
initial_data = observer(initial_query_points)


def build_svgp(data):
    variance = tf.math.reduce_variance(data.observations)
    kernel = gpflow.kernels.Matern52(variance=variance, lengthscales=[0.2, 0.2])
    likelihood = gpflow.likelihoods.Gaussian(variance=1e-5)
    svgp = gpflow.models.SVGP(
        kernel, likelihood, data.query_points, num_data=len(data.query_points)
    )
    gpflow.set_trainable(svgp.likelihood, False)

    return {
        "model": svgp,
        # "optimizer": tf.optimizers.Adam(learning_rate=0.01),
        # "optimizer_args": {
        #     "max_iter": 200,
        #     "batch_size": 10
        # }
    }


def build_sgpr(data):
    variance = tf.math.reduce_variance(data.observations)
    kernel = gpflow.kernels.Matern52(variance=variance, lengthscales=[0.2, 0.2])
    svgp = gpflow.models.SGPR(data.astuple(), kernel, data.query_points, noise_variance=1e-5)
    gpflow.set_trainable(svgp.likelihood, False)

    return {
        "model": svgp,
        "optimizer": tf.optimizers.Adam(learning_rate=0.01),
        "optimizer_args": {
            "max_iter": 200,
        },
    }


model = build_svgp(initial_data)

bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

result = bo.optimize(20, initial_data, model)
dataset = result.try_get_final_dataset()

query_points = dataset.query_points.numpy()
observations = dataset.observations.numpy()

arg_min_idx = tf.squeeze(tf.argmin(observations, axis=0))

print(f"query point: {query_points[arg_min_idx, :]}")
print(f"observation: {observations[arg_min_idx, :]}")

_, ax = plot_function_2d(
    branin, search_space.lower, search_space.upper, grid_density=30, contour=True
)
plot_bo_points(query_points, ax[0, 0], num_initial_points, arg_min_idx)
plt.show()
plt.close()

fig = plot_function_plotly(branin, search_space.lower, search_space.upper, grid_density=20)
fig.update_layout(height=500, width=500)

fig = add_bo_points_plotly(
    x=query_points[:, 0],
    y=query_points[:, 1],
    z=observations[:, 0],
    num_init=num_initial_points,
    idx_best=arg_min_idx,
    fig=fig,
)
fig.show()

_, ax = plt.subplots(1, 2)
plot_regret(observations, ax[0], num_init=num_initial_points, idx_best=arg_min_idx)
plot_bo_points(query_points, ax[1], num_init=num_initial_points, idx_best=arg_min_idx)

plt.show()
plt.close()

fig = plot_gp_plotly(
    result.try_get_final_model().model, search_space.lower, search_space.upper, grid_density=30
)

fig = add_bo_points_plotly(
    x=query_points[:, 0],
    y=query_points[:, 1],
    z=observations[:, 0],
    num_init=num_initial_points,
    idx_best=arg_min_idx,
    fig=fig,
    figrow=1,
    figcol=1,
)

fig.show()

gpflow.utilities.print_summary(result.try_get_final_model().model)

ls_list = [
    step.model.model.kernel.lengthscales.numpy()
    for step in result.history + [result.final_result.unwrap()]
]

ls = np.array(ls_list)
plt.plot(ls[:, 0])
plt.plot(ls[:, 1])

plt.show()
plt.close()
