# %% [markdown]
# # Bayesian optimization with deep neural networks and Thompson sampling

# %%
import numpy as np
import tensorflow as tf

np.random.seed(1793)
tf.random.set_seed(1793)


# %% [markdown]
# ## Define the problem and model
#
# Gaussian processes as a surrogate models are hard to beat on smaller datasets and optimization budgets. In some cases datasets might be larger, objective functions might be non-stationary or predictions need to be made quickly. Vanilla Gaussian processes scale poorly with amount of data, cannot easily capture nonstationarities and they are rather slow at prediction time. Here we show how recently developed uncertainty-aware neural networks can be used for Bayesian optimisation.
#
# In this example, we look to find the minimum value of the two-dimensional version of Michalewicz function over the hypercube $[0, \pi]^2$. We can represent the search space using a `Box`, and plot contours of the Michalewicz over this space. The Michalewicz function features sharp discountinuities that are difficult to capture with Gaussian processes.

# %%
import trieste
from math import pi
from trieste.utils.objectives import michalewicz
from util.plotting_plotly import plot_function_plotly

search_space = trieste.space.Box([0, 0], [pi, pi])

fig = plot_function_plotly(
    michalewicz, search_space.lower, search_space.upper, grid_density=100
)
fig.update_layout(height=1000, width=1000)
fig.show()


# The optimization procedure will benefit from having some starting data from the objective function to base its search on. We'll create an observer and evaluate it at 50 random points to generate an initial dataset.

# %%
from trieste.acquisition.rule import OBJECTIVE

num_initial_data_points = 50
initial_query_points = search_space.sample(num_initial_data_points)
observer = trieste.utils.objectives.mk_observer(michalewicz, OBJECTIVE)
initial_data = observer(initial_query_points)


# %% [markdown]
# The Bayesian optimization procedure estimates the next best points to query by using a probabilistic model of the objective. We'll use a neural network for this, provided by Keras. We will focus on ensembles of neural networks where predictions of a single network in the ensemble can be thought of as a sample from the posterior. This is also why Thompson sampling is a natural choice for the acquisition function with this type of model.

# %%
from trieste.models.keras.data import EnsembleDataTransformer
from trieste.models.optimizer import TFKerasOptimizer
from trieste.models.keras.models import NeuralNetworkEnsemble
from trieste.models.keras.networks import MultilayerFcNetwork, GaussianNetwork
from trieste.models.keras.utils import get_tensor_spec_from_data

ensemble_size = 20
input_tensor_spec, output_tensor_spec = get_tensor_spec_from_data(initial_data[OBJECTIVE])
networks = [
    GaussianNetwork(
        input_tensor_spec,
        output_tensor_spec,
        num_hidden_layers=2,
        units=[25, 25],
        activation=['relu', 'relu'],
    )
    for _ in range(ensemble_size)
]
optimizer = tf.keras.optimizers.Adam(0.1)
fit_args = {
    'batch_size': 16,
    'epochs': 40,
    'callbacks': [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20)],
    'validation_split': 0.1,
    'verbose': 0,
}
dataset_builder = EnsembleDataTransformer(networks, bootstrap_data=False)
model = NeuralNetworkEnsemble(
    networks,
    TFKerasOptimizer(optimizer, fit_args, dataset_builder),
    dataset_builder,
)


# %% [markdown]
# ## Create the Thompson sampling acquisition rule
#
# We achieve Bayesian optimization with Thompson sampling by specifying `ThompsonSampling` as the acquisition rule. Unlike in the usual `ThompsonSampling`, here the rule samples a model from the ensemble instead of directly from the posterior. It acquires `num_query_points` samples at `num_search_space_samples` points on the search space. It then returns the `num_query_points` points of those that minimise the model posterior.

# %%
acq_rule = trieste.acquisition.rule.ThompsonSampling(
    num_search_space_samples=1000, num_query_points=50
)

# %% [markdown]
# ## Run the optimization loop
#
# All that remains is to pass the Thompson sampling rule to the `BayesianOptimizer`. Once the optimization loop is complete, the optimizer will return `num_query_points` new query points for every step in the loop. With 3 steps, that's 150 points.

# %%
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

result = bo.optimize(3, initial_data, {OBJECTIVE: model}, acq_rule, track_state=False)
dataset = result.try_get_final_datasets()[OBJECTIVE]


# %% [markdown]
# ## Visualising the result
#
# We can take a look at where we queried the observer, both the original query points (crosses) and new query points (dots), and where they lie with respect to the contours of the Michalewicz.

# %%
from util.plotting import plot_function_2d, plot_bo_points

arg_min_idx = tf.squeeze(tf.argmin(dataset.observations, axis=0))
query_points = dataset.query_points.numpy()
observations = dataset.observations.numpy()
_, ax = plot_function_2d(
    michalewicz,
    search_space.lower,
    search_space.upper,
    grid_density=30,
    contour=True,
    figsize=(14,10),
)
plot_bo_points(query_points, ax[0, 0], num_initial_data_points, arg_min_idx)



# %% [markdown]
# We can also visualise the observations on a three-dimensional plot of the Michalewicz. We'll add the contours of the mean and variance of the model's predictive distribution as translucent surfaces.

# %%
from util.plotting_plotly import plot_keras_plotly, add_bo_points_plotly

fig = plot_keras_plotly(
    model,  # type: ignore
    search_space.lower,
    search_space.upper,
    grid_density=100
)
fig = add_bo_points_plotly(
    x=query_points[:, 0],
    y=query_points[:, 1],
    z=observations[:, 0],
    num_init=num_initial_data_points,
    idx_best=arg_min_idx,
    fig=fig,
    figrow=1,
    figcol=1,
)
fig.update_layout(height=1000, width=1000)
fig.show()

# %% [markdown]
# ## LICENSE
#
# [Apache License 2.0](https://github.com/secondmind-labs/trieste/blob/develop/LICENSE)
