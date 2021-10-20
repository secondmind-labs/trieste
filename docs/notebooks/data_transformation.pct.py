# %% [markdown]
# # Data transformation with the help of Ask-Tell interface.

# %%
import numpy as np
import gpflow
from gpflow.utilities import set_trainable
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

import trieste
from trieste.models.gpflow import GaussianProcessRegression
from trieste.models.optimizer import Optimizer
from trieste.objectives.utils import mk_observer
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.acquisition.function import ExpectedImprovement
from trieste.space import Box
from trieste.data import Dataset   
from trieste.ask_tell_optimization import AskTellOptimizer
from trieste.bayesian_optimizer import Record
from trieste.objectives.utils import mk_observer
from trieste.observer import OBJECTIVE

from util.plotting import plot_regret

np.random.seed(1794)
tf.random.set_seed(1794)


# %% [markdown]
# ## Describe the problem
#
# In this notebook, we show how to perform data transformation during Bayesian optimization. This is usually required by the models. A very common example is normalising the data before fitting the model, either min-max or standard normalization. This is usually done for numerical stability, or to improve or speed up the convergence. 
#
# In regression problem it is easy to perform data transformations as you do it once before the training. In Bayesian optimization this is more complex, as the data is added with each iteration and would need to be transformed again before the model is updated. At the moment Trieste cannot do such transformations for the user. Luckily, this can be easily done by using the [Ask-Tell interface](ask_tell_optimization.ipynb), as this is exactly the case where we want to have greater control of the optimization loop. The disadvantage is that it is up to the user to take care of all the data transformation. 
#
# As an example, we will be searching for a minimum of a 10-dimensional [Trid functions](https://www.sfu.ca/~ssurjano/trid.html). The range of variation of the Trid function values is large. It varies from values of $10^5$ to its global minimum $f(x^∗) = −210$. This large variation range makes it difficult for Bayesian optimization with Gaussian processes to find the global minimum. However, with data normalisation it becomes possible (see <cite data-cite="hebbal2019bayesian">[Hebbal et al. 2019](https://arxiv.org/abs/1905.03350)</cite>).

# %%
from trieste.objectives import (
    trid_10,
    TRID_10_MINIMUM,
    TRID_10_MINIMIZER,
    TRID_10_SEARCH_SPACE
)

function = trid_10
F_MINIMUM = TRID_10_MINIMUM
search_space = TRID_10_SEARCH_SPACE


# %% [markdown]
# ## Collect initial points 
#
# We set up the observer as usual over the Trid function search space, using Sobol sampling to sample the initial points.

# %%
num_initial_points = 50

observer = mk_observer(function)

initial_query_points = search_space.sample_sobol(num_initial_points)
initial_data = observer(initial_query_points)


# %% [markdown]
# ## Model the objective function
#
# The Bayesian optimization procedure estimates the next best points to query by using a probabilistic model of the objective. We'll use a Gaussian process (GP) model, built using GPflow. We also set priors over the hyperparameters.

# %%
def build_gp_model(data):
    variance = tf.math.reduce_variance(data.observations)
    kernel = gpflow.kernels.Matern52(variance=variance, lengthscales=[0.2]*data.query_points.shape[-1])
    prior_scale = tf.cast(1.0, dtype=tf.float64)
    kernel.variance.prior = tfp.distributions.LogNormal(tf.cast(-2.0, dtype=tf.float64), prior_scale)
    kernel.lengthscales.prior = tfp.distributions.LogNormal(tf.math.log(kernel.lengthscales), prior_scale)
    gpr = gpflow.models.GPR(data.astuple(), kernel, mean_function=gpflow.mean_functions.Constant(), noise_variance=1e-5)
    gpflow.set_trainable(gpr.likelihood, False)

    return GaussianProcessRegression(
        model=gpr,
        optimizer=Optimizer(
            gpflow.optimizers.Scipy(), 
            minimize_args={"options": dict(maxiter=100)}
        ),
        num_kernel_samples=100
    )

# build the model
model = build_gp_model(initial_data)



# %% [markdown]
# ## Run the optimization loop
#
# We can now run the Bayesian optimization loop by defining a `BayesianOptimizer` and calling its `optimize` method.
#
# The optimizer uses an acquisition rule to choose where in the search space to try on each optimization step. We'll be using Expected improvement.
#
# We'll run the optimizer for 100 steps. Note: this may take a while!

# %%
num_acquisitions = 100

bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
acquisition_rule = EfficientGlobalOptimization(ExpectedImprovement().using(OBJECTIVE))
result = bo.optimize(num_acquisitions, initial_data, model,
                         acquisition_rule=acquisition_rule, track_state=False)
dataset = result.try_get_final_dataset()


# %% [markdown]
# ## Explore the results
#
# We can now get the best point found by the optimizer. Note this isn't necessarily the point that was last evaluated. We will also plot regret for each optimization step.
#
# We can see that the optimization did not get close to the global optimum of -210.

# %%
query_points = dataset.query_points.numpy()
observations = dataset.observations.numpy()

arg_min_idx = tf.squeeze(tf.argmin(observations, axis=0))

print(f"query point: {query_points[arg_min_idx, :]}")
print(f"observation: {observations[arg_min_idx, :]}")


# %% [markdown]
# Once ask-tell optimization is over, you can extract an optimization result object and perform whatever analysis you need, just like with regular Trieste optimization interface. For instance, here we will plot regret for each optimization step.

# %%
def plot_ask_tell_regret(dataset):
    observations = dataset.observations.numpy()
    arg_min_idx = tf.squeeze(tf.argmin(observations, axis=0))

    suboptimality = observations - F_MINIMUM.numpy()
    ax = plt.gca()
    plot_regret(suboptimality, ax, num_init=num_initial_points, idx_best=arg_min_idx)

    ax.set_yscale("log")
    ax.set_ylabel("Regret")
    ax.set_ylim(0.001, 10000)
    ax.set_xlabel("# evaluations")

plot_ask_tell_regret(dataset)


# %% [markdown]
# # Data transformation with the help of Ask-Tell interface
#
# We first write a simple function for doing the standardisation of the data, that is, we scale the data to have a zero mean and a variance equal to 1. We also return the mean and standard deviation parameters as we will use them to transform new points.

# %%
def standardise(x, mean = None, std = None):
    if mean == None and std == None:
        mean = tf.math.reduce_mean(x, 0, True)
        std = tf.math.sqrt(tf.math.reduce_variance(x, 0, True))
    return (x - mean)/std, mean, std


# %% [markdown]
#
# Note that we also need to modify the search space, from the original $[-100, 100]$ for all 10 dimensions to the standardised space. For illustration, $[-1,1]$ will suffice here.

# %
search_space = Box([-1], [1])**10


# %% [markdown]
#
# Next we have to define our own Bayesian optimization loop where Ask-Tell optimizer performs optimisation, and we take care of data transformation and model fitting.
#
# We are using a simple approach whereby we normalize the initial data and use estimated mean and standard deviation from the initial normalization for transforming the new points that Bayesian optimization loop adds to the dataset. 

# %
x_sta, x_mean, x_std = standardise(initial_data.query_points) 
y_sta, y_mean, y_std = standardise(initial_data.observations) 
standardised_data = Dataset(
    query_points = x_sta, 
    observations = y_sta
)

dataset = initial_data
for step in range(num_acquisitions):

    # Retraining the model from scratch every 10 steps
    if step % 10 == 0:
        model = build_gp_model(standardised_data)
        model.optimize(standardised_data)
    else: 
        model.update(standardised_data)
        model.optimize(standardised_data)

    # Asking for a new point to observe
    ask_tell = AskTellOptimizer(search_space, standardised_data, model)
    query_point = ask_tell.ask()

    # Transforming the query point back to the non-standardised space
    query_point = x_std*query_point + x_mean

    # Evaluating the function at the new query point
    new_data_point = observer(query_point)
    dataset = dataset + new_data_point

    # Normalize the dataset with the new query point and observation 
    x_sta, _, _ = standardise(dataset.query_points, x_mean, x_std) 
    y_sta, _, _ = standardise(dataset.observations, y_mean, y_std) 
    standardised_data = Dataset(
        query_points = x_sta, 
        observations = y_sta
    )


# %% [markdown]
# ## Explore the results
#
# We inspect again the best point found by the optimizer and plot regret for each optimization step.
#
# We can see that the optimization now gets almost to the global optimum of -210.

# %%
query_points = dataset.query_points.numpy()
observations = dataset.observations.numpy()

arg_min_idx = tf.squeeze(tf.argmin(observations, axis=0))

plot_ask_tell_regret(dataset)

print(f"query point: {query_points[arg_min_idx, :]}")
print(f"observation: {observations[arg_min_idx, :]}")

#

# ## LICENSE
#
# [Apache License 2.0](https://github.com/secondmind-labs/trieste/blob/develop/LICENSE)
