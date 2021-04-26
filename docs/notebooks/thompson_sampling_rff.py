# %% [markdown]
# # Noise-free optimization with Expected Improvement

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
from trieste.utils.objectives import branin
from util.plotting_plotly import plot_function_plotly

search_space = trieste.space.Box([0, 0], [1, 1])

def noisy_branin(x):
    y = branin(x)
    return y + tf.random.normal(y.shape, stddev=0.1, dtype=y.dtype)

# fig = plot_function_plotly(branin, search_space.lower, search_space.upper, grid_density=20)
# fig.update_layout(height=400, width=400)
# fig.show()

# %% [markdown]
# ## Sample the observer over the search space
#
# Sometimes we don't have direct access to the objective function. We only have an observer that indirectly observes it. In _Trieste_, the observer outputs a number of datasets, each of which must be labelled so the optimization process knows which is which. In our case, we only have one dataset, the objective. We'll use _Trieste_'s default label for single-model setups, `OBJECTIVE`. We can convert a function with `branin`'s signature to a single-output observer using `mk_observer`.
#
# The optimization procedure will benefit from having some starting data from the objective function to base its search on. We sample five points from the search space and evaluate them on the observer.

# %%
from trieste.acquisition.rule import OBJECTIVE

observer = trieste.utils.objectives.mk_observer(noisy_branin, OBJECTIVE)

num_initial_points = 50
initial_query_points = search_space.sample(num_initial_points)
initial_data = observer(initial_query_points)

# %% [markdown]
# ## Model the objective function
#
# The Bayesian optimization procedure estimates the next best points to query by using a probabilistic model of the objective. We'll use Gaussian process regression for this, provided by GPflow. The model will need to be trained on each step as more points are evaluated, so we'll package it with GPflow's Scipy optimizer.
#
# Just like the data output by the observer, the optimization process assumes multiple models, so we'll need to label the model in the same way.

# %%
import gpflow
import gpflux
from trieste.utils.robustgp import ConditionalVariance
from trieste.utils.inducing_point_selectors import KMeans, GIBBON, UniformSampler, ConditionalVariance

num_data, input_dim = initial_data[OBJECTIVE].query_points.shape

from gpflow.config import default_float
from gpflux.layers.basis_functions.random_fourier_features import RandomFourierFeatures
from gpflux.sampling.kernel_with_feature_decomposition import KernelWithFeatureDecomposition


def build_rff_model(data, inducing_point_selector=KMeans) -> tf.keras.Model:
    var = tf.math.reduce_variance(data.observations)
    kernel = gpflow.kernels.SquaredExponential(variance=var, lengthscales=0.2 * np.ones(2, ))
    num_rff = 10000
    features = RandomFourierFeatures(kernel, num_rff, dtype=default_float())
    coefficients = np.ones((num_rff, 1), dtype=default_float())
    kernel_with_features = KernelWithFeatureDecomposition(kernel, features, coefficients)

    num_inducing = 40
    Z = inducing_point_selector(data.query_points,data.observations,num_inducing, kernel).get_points()
    inducing_variable = gpflow.inducing_variables.InducingPoints(Z)
    gpflow.utilities.set_trainable(inducing_variable, False)

    layer = gpflux.layers.GPLayer(
        kernel_with_features,
        inducing_variable,
        num_data,
        whiten=False,
        num_latent_gps=1,
        mean_function=gpflow.mean_functions.Zero(),
    )
    likelihood = gpflow.likelihoods.Gaussian(1e-0)
    gpflow.utilities.set_trainable(likelihood, True)
    likelihood_layer = gpflux.layers.LikelihoodLayer(likelihood)
    model = gpflux.models.DeepGP([layer], likelihood_layer)
    return model, inducing_point_selector


# %%
from trieste.models.model_interfaces import GPFluxModel

model, inducing_point_selector = build_rff_model(initial_data[OBJECTIVE], GIBBON)
model = GPFluxModel(model, initial_data[OBJECTIVE], num_epochs=10000, batch_size=500, inducing_point_selector=inducing_point_selector)
model.optimize(initial_data[OBJECTIVE])
models = {OBJECTIVE: model}

# %% [markdown]
# ## Run the optimization loop

# %%
neg_traj = trieste.acquisition.NegativeGaussianProcessTrajectory()
rule = trieste.acquisition.rule.BatchByMultipleFunctions(neg_traj.using(OBJECTIVE), num_query_points=5)
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
result = bo.optimize(10, initial_data, models, acquisition_rule=rule, track_state=False)
dataset = result.try_get_final_datasets()[OBJECTIVE]

# %% [markdown]
# ## Explore the results

# %%
query_points = dataset.query_points.numpy()
observations = dataset.observations.numpy()

arg_min_idx = tf.squeeze(tf.argmin(observations, axis=0))

print(f"query point: {query_points[arg_min_idx, :]}")
print(f"observation: {observations[arg_min_idx, :]}")

# %% [markdown]
# We can visualise how the optimizer performed by plotting all the acquired observations, along with the true function values and optima, either in a two-dimensional contour plot ...

# %%
from util.plotting import plot_function_2d, plot_bo_points

_, ax = plot_function_2d(
    branin, search_space.lower, search_space.upper, grid_density=30, contour=True
)
plot_bo_points(query_points, ax[0, 0], num_initial_points, arg_min_idx)

# %% [markdown]
# ... or as a three-dimensional plot

# %%
from util.plotting_plotly import add_bo_points_plotly
from util.plotting_plotly import add_gp_trajectories

fig = plot_function_plotly(branin, search_space.lower, search_space.upper,
                           grid_density=40)
fig.update_layout(scene_aspectmode='cube')

fig = add_bo_points_plotly(
    x=query_points[:, 0],
    y=query_points[:, 1],
    z=observations[:, 0],
    num_init=num_initial_points,
    idx_best=arg_min_idx,
    fig=fig,
)
fig.show()

from util.plotting_plotly import plot_gp_plotly, add_bo_points_plotly

# fig = plot_gp_plotly(
#     result.try_get_final_models()[OBJECTIVE],
#     search_space.lower,
#     search_space.upper,
#     grid_density=30
# )
fig = plot_function_plotly(branin, search_space.lower, search_space.upper, grid_density=20)

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
fig = add_gp_trajectories(fig=fig,
                          model=result.try_get_final_models()[OBJECTIVE],
                          mins=search_space.lower,
                          maxs=search_space.upper,
                          grid_density=40,
                          ntraj=10)
fig.update_layout(scene_aspectmode='cube')
fig.write_html("trajectories.html")
fig.show()

from util.plotting_plotly import plot_gp_plotly, add_bo_points_plotly

fig = plot_gp_plotly(
    result.try_get_final_models()[OBJECTIVE],
    search_space.lower,
    search_space.upper,
    grid_density=40
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
fig.update_layout(scene_aspectmode='cube')
fig.show()

# %% [markdown]
# We can also visualise the how each successive point compares the current best.
#
# We produce two plots. The left hand plot shows the observations (crosses and dots), the current best (orange line), and the start of the optimization loop (blue line). The right hand plot is the same as the previous two-dimensional contour plot, but without the resulting observations. The best point is shown in each (purple dot).

# %%
import matplotlib.pyplot as plt
from util.plotting import plot_regret

fig, ax = plt.subplots(1, 2)
plot_regret(observations, ax[0], num_init=num_initial_points, idx_best=arg_min_idx)
plot_bo_points(query_points, ax[1], num_init=num_initial_points, idx_best=arg_min_idx)
fig.show()

# %% [markdown]
# ## LICENSE
#
# [Apache License 2.0](https://github.com/secondmind-labs/trieste/blob/develop/LICENSE)
