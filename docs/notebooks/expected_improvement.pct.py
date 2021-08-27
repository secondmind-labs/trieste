# %% [markdown]
# # Noise-free optimization with Expected Improvement

# %%
import numpy as np
import tensorflow as tf
import trieste
import gpflow
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

from trieste.objectives import logarithmic_goldstein_price, LOGARITHMIC_GOLDSTEIN_PRICE_MINIMUM
from util.plotting_plotly import plot_function_plotly
from trieste.space import Box
from trieste.acquisition.rule import TrustRegion
from util.plotting import plot_bo_points, plot_function_2d

np.random.seed(179)
tf.random.set_seed(179)
search_space = Box([0, 0], [1, 1])

observer = trieste.objectives.utils.mk_observer(logarithmic_goldstein_price)

num_initial_points = 5
initial_query_points = search_space.sample_sobol(num_initial_points)
initial_data = observer(initial_query_points)


def build_model(data):
    variance = tf.math.reduce_variance(data.observations)
    kernel = gpflow.kernels.Matern52(variance=variance, lengthscales=[0.2, 0.2])
    prior_scale = tf.cast(1.0, dtype=tf.float64)
    kernel.variance.prior = tfp.distributions.LogNormal(tf.cast(0.0, dtype=tf.float64), prior_scale)
    kernel.lengthscales.prior = tfp.distributions.LogNormal(tf.math.log(kernel.lengthscales), prior_scale)
    gpr = gpflow.models.GPR(data.astuple(), kernel, noise_variance=1e-5)
    gpflow.set_trainable(gpr.likelihood, False)

    return {
            "model": gpr,
            "model_args": {
                "num_kernel_samples": 100,
            },
            "optimizer": gpflow.optimizers.Scipy(),
            "optimizer_args": {
                "minimize_args": {"options": dict(maxiter=100)},
            },
    }

model = build_model(initial_data)


def plot_rectangle(space, ax):
    lx = space.lower[0].numpy()
    ly = space.lower[1].numpy()
    ux = space.upper[0].numpy()
    uy = space.upper[1].numpy()
    ax.plot([lx, lx, ux, ux, lx], [ly, uy, uy, ly, ly])


rule = TrustRegion(beta = 0.5)
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
acq_state = None
dataset = initial_data
for i in range(4):
    result = bo.optimize(1, dataset, model, acquisition_rule=rule, acquisition_state=acq_state)
    acq_state = result.final_result.unwrap().acquisition_state
    dataset = result.try_get_final_dataset()
    arg_min_idx = tf.squeeze(tf.argmin(dataset.observations, axis=0))
    _, ax = plot_function_2d(
        logarithmic_goldstein_price, search_space.lower, search_space.upper, grid_density=40, contour=True
    )

    plot_bo_points(
        dataset.query_points.numpy(),
        ax=ax[0, 0],
        num_init=dataset.observations.shape[0] -1,
        idx_best=arg_min_idx,
    )
    plot_rectangle(acq_state.acquisition_space, ax[0, 0])

    plt.savefig('fig/fig{:02d}.png'.format(i))
    plt.close()