# %% [markdown]
# # Bayesian active learning of failure or feasibility regions 
# 
# When designing a system it is important to identify design parameters that may affect the reliability of the system and cause failures, or lead to unsatisfactory performance. Consider designing a communication network that for some design parameters would lead to too long delays for users. A designer of the system would then decide what is the maximum acceptable delay and want to identify a *failure region* in the parameter space that would lead to longer delays., or conversely, a *feasible region* with safe performance. 
#
# When evaluating the system is expensive (e.g. lengthy computer simulations), identification of the failure region needs to be performed with a limited number of evaluations. Traditional Monte Carlo based methods are not suitable here as they require too many evaluations. Bayesian active learning methods, however, are well suited for the task. Here we show how Trieste can be used to identify failure or feasible regions with the help of acquisition functions designed with this goal in mind. 
#

# %%
# %matplotlib inline

# silence TF warnings and info messages, only print errors
# https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

import numpy as np

np.random.seed(1793)
tf.random.set_seed(1793)


# %% [markdown]
# ## A toy problem
#
# Throughout the tutorial we will use the standard Branin function as a stand-in for an expensive-to-evaluate system. We create a failure region by thresholding the value at 80, space with value above 80 is considered a failure region. This region needs to be learned as efficiently as possible by the active learning algorithm.
#
# Note that if we are interested in a feasibility region instead, it is simply a complement of the failure region, space with the value below 80.
#
# We illustrate the thresholded Branin function below, you can note that above the threshold of 80 there are no more values observed.

# %%
from trieste.objectives import branin, BRANIN_SEARCH_SPACE
from util.plotting_plotly import plot_function_plotly
from trieste.space import Box

search_space = BRANIN_SEARCH_SPACE

# threshold is arbitrary, but has to be within the range of the function
threshold = 80

# define a modified branin function
def thresholded_branin(x):
    y = np.array(branin(x))
    y[y > threshold] = np.nan
    return tf.convert_to_tensor(y.reshape(-1, 1), x.dtype)

# illustrate the thresholded branin function
fig = plot_function_plotly(
    thresholded_branin, search_space.lower, search_space.upper, grid_density=700
)
fig.update_layout(height=800, width=800)
fig.show()


# %% [markdown]
# We start with a small initial dataset where our expensive-to-evaluate function is evaluated on points coming from a space-filling Halton sequence.

# %%
import trieste

observer = trieste.objectives.utils.mk_observer(branin)

num_initial_points = 12
initial_query_points = search_space.sample_halton(num_initial_points)
initial_data = observer(initial_query_points)


# %% [markdown]
# ## Probabilistic model of the objective function
#
# Just like in sequential optimization, we use a probabilistic model of the objective function. Acquisition functions will exploit the predictive posterior of the model to identify the failure region. We use a `GPR` model from the GPflow library to formulate a Gaussian process model, wrapping it in Trieste's `GaussianProcessRegression` model wrapper. As a good practice, we use priors for the kernel hyperparameters.

# %%
import gpflow
from trieste.models.gpflow.models import GaussianProcessRegression
import tensorflow_probability as tfp

def build_model(data):
    variance = tf.math.reduce_variance(data.observations)
    kernel = gpflow.kernels.Matern52(variance=variance, lengthscales=[0.2, 0.2])
    prior_scale = tf.cast(1.0, dtype=tf.float64)
    kernel.variance.prior = tfp.distributions.LogNormal(tf.cast(-2.0, dtype=tf.float64), prior_scale)
    kernel.lengthscales.prior = tfp.distributions.LogNormal(tf.math.log(kernel.lengthscales), prior_scale)
    gpr = gpflow.models.GPR(data.astuple(), kernel, noise_variance=1e-5)
    gpflow.set_trainable(gpr.likelihood, False)

    return GaussianProcessRegression(gpr)


model = build_model(initial_data)


# %% [markdown]
# ## Active learning with Expected feasibility acquisition function
#
# The problem of identifying a failure or feasibility region of a (expensive-to-evaluate) function $f$ can be formalized as estimating the excursion set, $\Gamma^* = \{ x \in X: f(x) \ge T\}$, or estimating the contour line, $C^* = \{ x \in X: f(x) = T\}$, for some threshold $T$ (see <cite data-cite="bect2012sequential"/> for more details). 
#
# It turns out that Gaussian processes can be used as classifiers for identifying where excursion probability is larger than 1/2 and this idea is used to build many sequential sampling strategies. Here we introduce Expected feasibility acquisition function that implements two related sampling strategies called *bichon* criterion (<cite data-cite="bichon2008efficient"/>) and *ranjan* criterion (<cite data-cite="ranjan2008sequential"/>). <cite data-cite="bect2012sequential"/> provides a common expression for these two criteria: $$\mathbb{E}[\max(0, (\alpha s(x))^\delta - |T - m(x)|^\delta)]$$
#
# Here $m(x)$ and $s(x)$ are the mean and standard deviation of the predictive posterior of the Gaussian process model. Bichon criterion is obtained when $\delta = 1$ while ranjan criterion is obtained when $\delta = 2$. $\alpha>0$ is another parameter that acts as a percentage of standard deviation of the posterior around the current boundary estimate where we want to sample. The goal is to sample a point with a mean close to the threshold $T$ and a high variance, so that the positive difference in the equation above is as large as possible.


# %% [markdown]
# We now illustrate `ExpectedFeasibility` acquisition function using the Bichon criterion. Performance for the Ranjan criterion is typically very similar. `ExpectedFeasibility` takes threshold as an input and has two parameters, `alpha` and `delta` following the description above. 
#
# Note that even though we use below `ExpectedFeasibility` with `EfficientGlobalOptimization`  `BayesianOptimizer` routine, we are actually performing active learning. The only relevant difference between the two is the nature of the acquisition function - optimization ones are designed with the goal of finding the optimum of a function, while active learning ones are designed to learn the function (or some aspect of it, like here).

# %%
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.acquisition.function import ExpectedFeasibility

# Bichon criterion
delta = 1

# set up the acquisition rule and initialize the Bayesian optimizer
acq = ExpectedFeasibility(threshold, delta=delta)
rule = EfficientGlobalOptimization(builder=acq)  # type: ignore
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

num_steps = 10
result = bo.optimize(num_steps, initial_data, model, rule)


# %% [markdown]
# Let's illustrate the results.
#
# To identify the failure or feasibility region we compute the excursion probability using our Gaussian process model: $$P\left(\frac{m(x) - T}{s(x)}\right)$$ where $m(x)$ and $s(x)$ are the mean and standard deviation of the predictive posterior of the model given the data. 
#
# We plot a two-dimensional contour map of our thresholded Branin function as a reference, excursion probability map using the model fitted to the initial data alone, and updated excursion probability map after all the active learning steps.
#
# We first define helper functions for computing excursion probabilities and plotting, and then plot the thresholded Branin function as a reference. White area represents the failure region.

# %%
from util.plotting import plot_bo_points, plot_function_2d


def excursion_probability(x, model, threshold=80):
    mean, variance = model.model.predict_f(x)
    normal = tfp.distributions.Normal(tf.cast(0, x.dtype), tf.cast(1, x.dtype))
    t = (mean - threshold) / tf.sqrt(variance)
    return normal.cdf(t)


def plot_excursion_probability(title, model = None, query_points = None):
    
    if model is None:
        objective_function = thresholded_branin
    else:
        def objective_function(x):
            return excursion_probability(x, model)

    _, ax = plot_function_2d(
        objective_function,
        search_space.lower - 0.01,
        search_space.upper + 0.01,
        grid_density=300,
        contour=True,
        colorbar=True,
        figsize=(10, 6),
        title=[title],
        xlabel="$X_1$",
        ylabel="$X_2$",
    )
    if query_points is not None:
        plot_bo_points(query_points, ax[0, 0], num_initial_points)

plot_excursion_probability("Excursion set, Branin function")


# %% [markdown]
# Next we illustrate the excursion probability map using the model fitted to the initial data alone. On the figure below we can see that the failure region boundary has been identified with some accuracy in the upper right corner, but not in the lower left corner. It is also loosely defined, as indicated by a slow decrease and increase away from the 0.5 excursion probability contour.

# %%
# extracting the data to illustrate the points
dataset = result.try_get_final_dataset()
query_points = dataset.query_points.numpy()
observations = dataset.observations.numpy()

# fitting the model only to the initial data
initial_model = build_model(initial_data)
initial_model.optimize(initial_data)

plot_excursion_probability(
    "Probability of excursion, initial data", initial_model, query_points[:num_initial_points,]
)


# %% [markdown]
# Next we examine an updated excursion probability map after the 10 active learning steps. We can now see that the model is much more accurate and confident, as indicated by a good match with the reference thresholded Branin function and sharp decrease/increase away from the 0.5 excursion probability contour.

# %%
updated_model = result.history[-1].models["OBJECTIVE"]

plot_excursion_probability(
    "Updated probability of excursion", updated_model, query_points
)


# %% [markdown]
# We can also examine what would happen if we would continue for many more active learning steps. One would expect that choices would be allocated closer and closer to the boundary, and uncertainty continuing to collapse. Indeed, on the figure below we observe exactly that. With 90 observations more the model is precisely representing the failure region boundary. It is somewhat difficult to see on the figure, but the most of the additional query points lie exactly on the threshold line. 

# %%
num_steps = 90
result = bo.optimize(num_steps, dataset, model, rule)

final_model = result.history[-1].models["OBJECTIVE"]
dataset = result.try_get_final_dataset()
query_points = dataset.query_points.numpy()

plot_excursion_probability("Final probability of excursion", final_model, query_points)


# %% [markdown]
# ## LICENSE
#
# [Apache License 2.0](https://github.com/secondmind-labs/trieste/blob/develop/LICENSE)
