# %% [markdown]
# # S-GP-TS Demo

# %% [markdown]
# This code accomapnies the paper "Scalable Thompson Sampling usingSparse Gaussian Process Models".
#
# First we demonstrate the method on a simple 2D benchmark, before showing how more complicated experiments can be ran.

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import numpy as np
import tensorflow as tf

np.random.seed(1789)
tf.random.set_seed(1789)
tf.keras.backend.set_floatx("float64")

# %% [markdown]
# ## Describe the problem
# In this example, we look to find the minimum value of the two-dimensional Branin function over the hypercube $[0, 1]^2$. We can represent the search space using a `Box`, and plot contours of the Branin over this space.

# %%
import trieste
from trieste.objectives import scaled_branin, SCALED_BRANIN_MINIMUM
from trieste.objectives.utils import mk_observer
from util.plotting_plotly import plot_function_plotly
from trieste.utils.inducing_point_selectors import KMeans, ConditionalVariance
import tensorflow_probability as tfp

search_space = trieste.space.Box([0, 0], [1, 1])
noise = 0.001


def noisy_branin(x):
    y = scaled_branin(x)
    return y + tf.random.normal(y.shape, stddev=noise * tf.reduce_sum(x, axis=-1, keepdims=True), dtype=y.dtype)


fig = plot_function_plotly(scaled_branin, search_space.lower, search_space.upper, grid_density=20)
fig.update_layout(height=400, width=400)
fig.show()

# %% [markdown]
# ## Sample the observer over the search space
#
# In _Trieste_, the observer outputs a number of datasets, each of which must be labelled so the optimization process knows which is which. In our case, we only have one dataset, the objective. We'll use _Trieste_'s default label for single-model setups, `OBJECTIVE`. We can convert a function with `branin`'s signature to a single-output observer using `mk_observer`.
#
# The optimization procedure will benefit from having some starting data from the objective function to base its search on. We sample five points from the search space and evaluate them on the observer.

# %%
from trieste.acquisition.rule import OBJECTIVE

observer = mk_observer(noisy_branin, OBJECTIVE)

num_initial_points = 200
initial_query_points = search_space.sample(num_initial_points)
initial_data = observer(initial_query_points)

# %% [markdown]
# ## Model the objective function
#
# The Bayesian optimization procedure estimates the next best points to query by using a probabilistic model of the objective. We'll use a Sparse Variational Gaussian process for this, as provided by GPflux. The model will need to be trained on each step as more points are evaluated.
#
# We use a K-means clustering routine to choose 10 inducing points upon which we base our variational approximation.
#
#
# Just like the data output by the observer, the optimization process assumes multiple models, so we'll need to label the model in the same way.

# %%
import gpflow
import gpflux
from gpflow.config import default_float
from gpflux.layers.basis_functions.random_fourier_features import RandomFourierFeatures
from gpflux.sampling.kernel_with_feature_decomposition import KernelWithFeatureDecomposition
from trieste.models.gpflux.models import FeaturedHetGPFluxModel
from gpflux.helpers import construct_basic_inducing_variables
num_data, input_dim = initial_data[OBJECTIVE].query_points.shape
num_inducing_points = 20

inducing_point_selector = KMeans(search_space)


def create_kernel_with_features(var, input_dim):
    num_rff = 1000
    kernel = gpflow.kernels.SquaredExponential(variance=var, lengthscales=0.2 * np.ones(input_dim, ))
    coefficients = np.ones((num_rff, 1), dtype=default_float())
    features = RandomFourierFeatures(kernel, num_rff, dtype=default_float())
    return KernelWithFeatureDecomposition(kernel, features, coefficients)

def build_rff_model(data):
    var = tf.math.reduce_variance(data.observations)

    kernel_with_features1 =create_kernel_with_features(var, input_dim)
    kernel_with_features2 = create_kernel_with_features(var / 2., input_dim)

    kernel_list = [kernel_with_features1, kernel_with_features2]
    kernel = gpflux.helpers.construct_basic_kernel(kernel_list)

    Z = inducing_point_selector.get_points(data.query_points, data.observations,
                                           num_inducing_points, kernel, noise)

    inducing_variable = construct_basic_inducing_variables(num_inducing_points, input_dim,
    output_dim=2, share_variables=True, z_init= Z)
    gpflow.utilities.set_trainable(inducing_variable, False)

    layer = gpflux.layers.GPLayer(
        kernel,
        inducing_variable,
        num_data,
        whiten=False,
        num_latent_gps=2,
        mean_function=gpflow.mean_functions.Constant(),
    )
    layer.returns_samples = False

    likelihood = gpflow.likelihoods.HeteroskedasticTFPConditional(
        distribution_class=tfp.distributions.Normal,
        scale_transform=tfp.bijectors.Exp(),
    )

    likelihood_layer = gpflux.layers.LikelihoodLayer(likelihood)
    model = gpflux.models.DeepGP([layer], likelihood_layer)

    epochs = 10000
    batch_size = 200
    optimizer = tf.optimizers.Adam(0.05)

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss", patience=10, factor=0.5, verbose=1, min_lr=1e-6,
        ),
        tf.keras.callbacks.EarlyStopping(monitor="loss", patience=50, min_delta=0.01, verbose=1,
                                         mode="min"),

    ]

    fit_args = {
        "batch_size": batch_size,
        "epochs": epochs,
        "verbose": 2,
        "callbacks": callbacks,
    }

    return FeaturedHetGPFluxModel(model=model, optimizer=optimizer, fit_args=fit_args)


model = build_rff_model(initial_data[OBJECTIVE])
model.optimize(initial_data[OBJECTIVE])
models = {OBJECTIVE: model}

from util.plotting import plot_gp_2d

fig, ax = plot_gp_2d(model.model_gpflux, search_space.lower,
    search_space.upper,
    grid_density=30)

fig.axes[0].scatter(initial_query_points[:, 0], initial_query_points[:, 1],
                    initial_data[OBJECTIVE].observations.numpy())

Z = model.model_gpflux.f_layers[0].inducing_variable.inducing_variable.Z

q_mu = model.model_gpflux.f_layers[0].q_mu
q_sqrt = model.model_gpflux.f_layers[0].q_sqrt

# import matplotlib.pyplot as plt
# plt.figure()
# plt.scatter(Z[:,0], Z[:, 1])

# model.model_gpflux
###################################################
###################################################
###################################################
# # %% [markdown]
# # ## Run the optimization loop
#
# # %% [markdown]
# # We run 5 BO iterations, each recommending a batch of 25 locations.
#
# # %%
# neg_traj = trieste.acquisition.NegativeGaussianProcessTrajectory()
# rule = trieste.acquisition.rule.EfficientGlobalOptimization(neg_traj.using(OBJECTIVE), num_query_points=20)
#
# bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
# result = bo.optimize(5, initial_data, models, acquisition_rule=rule, track_state=False)
#
# # %% [markdown]
# # ## Explore the results
#
# # %%
# dataset = result.try_get_final_datasets()[OBJECTIVE]
# query_points = dataset.query_points.numpy()
# observations = dataset.observations.numpy()
# true_scores = scaled_branin(dataset.query_points).numpy()
# arg_min_idx = tf.squeeze(tf.argmin(true_scores, axis=0))
#
# print(f"Believed optima: {query_points[arg_min_idx, :]}")
# print(f"Objective function value: {true_scores[arg_min_idx, :]}")
#
# # %% [markdown]
# # We can visualise how the optimizer performed by plotting all the acquired observations (green dots), along with the initial design (green crosses) and the true function (contours). We see that S-GP-TS is able to focus resources into making evaluations in promising areas of the space.
#
# # %%
# from util.plotting import plot_function_2d, plot_bo_points
#
# _, ax = plot_function_2d(
#     scaled_branin, search_space.lower, search_space.upper, grid_density=30, contour=True
# )
# plot_bo_points(query_points, ax[0, 0], num_initial_points, arg_min_idx)
#
# # %% [markdown]
# # We can also visualise the how each successive point compares the current best.
# #
# # We produce two plots. The left hand plot shows the observations (crosses and dots), the current best (orange line), and the start of the optimization loop (blue line). The right hand plot is the same as the previous two-dimensional contour plot, but without the contours. The best point is shown in each (purple dot).
#
# # %%
# import matplotlib.pyplot as plt
# from util.plotting import plot_regret
# from util.plotting_plotly import add_bo_points_plotly
#
# fig, ax = plt.subplots(1, 2)
# plot_regret(true_scores - SCALED_BRANIN_MINIMUM.numpy(), ax[0], num_init=num_initial_points, idx_best=arg_min_idx)
# ax[0].set_ylim(0.00001, 1000)
# ax[0].set_yscale("log")
# plot_bo_points(query_points, ax[1], num_init=num_initial_points, idx_best=arg_min_idx)
# fig.show()
#
# from util.plotting import plot_gp_2d
#
# plot_gp_2d(result.try_get_final_model().model_gpflux, search_space.lower,
#     search_space.upper,
#     grid_density=30)
###################################################
###################################################
###################################################
#
# from util.plotting_plotly import plot_gp_plotly
#
# fig = plot_gp_plotly(
#     result.try_get_final_model().model_gpflux,  # type: ignore
#     search_space.lower,
#     search_space.upper,
#     grid_density=30,
# )
#
# fig = add_bo_points_plotly(
#     x=query_points[:, 0],
#     y=query_points[:, 1],
#     z=observations[:, 0],
#     num_init=num_initial_points,
#     idx_best=arg_min_idx,
#     fig=fig,
#     figrow=1,
#     figcol=1,
# )
# fig.show()
#
# # %% [markdown]
# # # Full Synthetic Experiments
#
# # %% [markdown]
# # We now show how to run S-GP-TS on the synthetic benchmarks, as demonstrated in the paper. We focus on the hartmann function, but our other benchmarks can be ran similarly (see commented out code).
#
# # %% [markdown]
# # First we prepare the problem
#
# # %%
# from trieste.utils.objectives import hartmann_6
#
# # from trieste.utils.objectives import ackley_5
# # from trieste.utils.objectives import shekel_4
#
#
# search_space = trieste.space.Box([0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1])
# noise = 0.1
#
# exact_objective = hartmann_6
# noise = tf.cast(0.5, dtype=tf.float64)
#
#
# def noisy_hartmann_6(x):
#     y = exact_objective(x)
#     return y + tf.random.normal(y.shape, stddev=tf.math.sqrt(noise), dtype=y.dtype)
#
#
# observer = trieste.utils.objectives.mk_observer(noisy_hartmann_6, OBJECTIVE)
#
# num_initial_points = 250
# initial_query_points = search_space.sample(num_initial_points)
# initial_data = observer(initial_query_points)
# num_data, input_dim = initial_data[OBJECTIVE].query_points.shape
#
# # %% [markdown]
# # We then choose our inducing point selection strategy and the number of considered inducing points.
#
# # %%
# inducing_point_selector = KMeans(search_space)
# # inducing_point_selector = ConditionalVariance(search_space)
# num_inducing_points = 100
#
# # %% [markdown]
# # Prepare the S-GP-TS model.
#
# # %%
# model, inducing_point_selector = build_rff_model(initial_data[OBJECTIVE], inducing_point_selector)
# model = GPFluxModel(model, initial_data[OBJECTIVE], num_epochs=10000, batch_size=100,
#                     inducing_point_selector=inducing_point_selector, max_num_inducing_points=num_inducing_points)
# model.optimize(initial_data[OBJECTIVE])
# models = {OBJECTIVE: model}
#
# # %% [markdown]
# # Run the experiment for 40 iterations with batches of size 100.
#
# # %%
# neg_traj = trieste.acquisition.NegativeGaussianProcessTrajectory()
# rule = trieste.acquisition.rule.BatchByMultipleFunctions(neg_traj.using(OBJECTIVE), num_query_points=100)
# bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
# result = bo.optimize(40, initial_data, models, acquisition_rule=rule, track_state=False)[0]
#
# # %% [markdown]
# # ## LICENSE
# #
# # [Apache License 2.0](https://github.com/secondmind-labs/trieste/blob/develop/LICENSE)
