# %% [markdown]
# # Building models with configuration dictionaries
#
# If you are an expert user of Trieste and some modelling library, GPflow for example, then building models via a configuration dictionary might be a useful alternative to working with model and optimizer wrappers. Here we provide an overview of how to use configuration dictionaries.

# %%
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import gpflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tf.get_logger().setLevel("ERROR")

import trieste
from trieste.objectives import BRANIN_SEARCH_SPACE, SCALED_BRANIN_MINIMUM, scaled_branin
from trieste.objectives.utils import mk_observer
from trieste.space import Box

np.random.seed(1793)
tf.random.set_seed(1793)


# %% [markdown]
# ## Finding a minimum of the Branin function
#
# In this example, as in many other tutorials, we look to find the minimum value of the familiar two-dimensional Branin function over the hypercube $[0, 1]^2$.

# %%
# convert the objective function to a single-output observer
observer = trieste.objectives.utils.mk_observer(scaled_branin)

# Sample the observer over the search space
num_initial_points = 5
search_space = BRANIN_SEARCH_SPACE
initial_query_points = search_space.sample_sobol(num_initial_points)
initial_data = observer(initial_query_points)


# %% [markdown]
# ## Standard way of setting up a model of the objective function
#
# The Bayesian optimization procedure estimates the next best points to query by using a probabilistic model of the objective. We'll use Gaussian Process (GP) regression in this tutorial, as provided by GPflow.
#
# The GPflow models cannot be used directly in our Bayesian optimization routines, only through a valid model wrapper. Trieste has wrappers that support several popular models. For instance, `GPR` and `SGPR` models from GPflow have to be used with the `GaussianProcessRegression` wrapper. These wrappers standardise outputs from all models, deal with preparation of the data and implement additional methods needed for Bayesian optimization.
#
# Typical process of setting up a valid model would go as follow.  We first set up a GPR model, using some initial data to set some parameters.

# %%
from gpflow.models import GPR

from trieste.models.gpflow import GaussianProcessRegression
from trieste.models.optimizer import Optimizer


def build_model(data):
    variance = tf.math.reduce_variance(data.observations)
    kernel = gpflow.kernels.Matern52(variance=variance, lengthscales=[0.2, 0.2])
    gpr = gpflow.models.GPR(data.astuple(), kernel, noise_variance=1e-5)
    return gpr


gpflow_model = build_model(initial_data)

# %% [markdown]
# Usually constructing a GPflow model would be enough, as it is the only required argument for the model wrappers. Wrappers have other arguments — an `optimizer` argument as a rule and potentially some additional model arguments (for example, `num_kernel_samples` in `GaussianProcessRegression`). These arguments are set to sensible defaults and hence typically we can simplify the model building. 

# %%
model = GaussianProcessRegression(gpflow_model)

# %% [markdown]
# However, as expert users, we might want to customize the optimizer for the model and set some arguments that we want to pass to it. We need to use Trieste's optimizer wrappers for that; here `Optimizer` would be the suitable wrapper. We'll optimize our model with GPflow's Scipy optimizer and pass some custom parameters to it.

# %%
optimizer = Optimizer(
    optimizer=gpflow.optimizers.Scipy(), minimize_args={"options": dict(maxiter=100)}
)

# %% [markdown]
# Finally we build a valid model that can be used with `BayesianOptimizer`. For the `GPR` model we need to use the `GaussianProcessRegression` wrapper. We also set a wrapper specific parameter for initialising the kernel.

# %%
model = GaussianProcessRegression(gpflow_model, optimizer=optimizer, num_kernel_samples=100)

# %% [markdown]
# We can now run the Bayesian optimization loop by defining a `BayesianOptimizer` and calling its `optimize` method. We are not interested in results here, but for the sake of completeness, lets run the Bayesian optimization as well.

# %%
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
result = bo.optimize(2, initial_data, model)


# %% [markdown]
# ## Using configuration dictionaries
#
# Instead of working directly with model and optimizer wrappers, after you know them sufficiently well, you can skip them by using configuration dictionary. It consists of a dictionary with same four arguments that can be passed to any model wrapper: `model`, `model_args`, `optimizer` and `optimizer_args`.
#
# In the background Trieste combines the `optimizer` and `optimizer_args` to build an optimizer wrapper and then combines the `model`, `model_args` and optimizer wrapper to build a model using the appropriate model wrapper.
#
# Let's see this in action. We will re-use the `GPR` model we have created above and use the same additional arguments. As you can see, you retain all the flexibility but can skip working with the interfaces if you know them well already.

# %%
model_config = {
    "model": gpflow_model,
    "model_args": {
        "num_kernel_samples": 100,
    },
    "optimizer": gpflow.optimizers.Scipy(),
    "optimizer_args": {
        "minimize_args": {"options": dict(maxiter=100)},
    },
}

# %% [markdown]
# Next you simply pass the configuration dictionary to the `optimize` function and `BayesianOptimizer` will sort out which model and optimizer wrapper needs to be used to build a valid model.

# %%
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
result = bo.optimize(2, initial_data, model_config)


# %% [markdown]
# ## Using configuration dictionaries for setting up experiments
#
# Another use case is in setting up experiments, where it becomes easier to benchmark Bayesian optimization algorithms. The advantage is that we can easily change the models and set up any argument for them from one experiment to another. We only need to change the object with the experiment specification (`experiment_conditions` below), while the rest of the code for executing experiments can stay the same. Below is an illustration of how could that look like.

# %%
from copy import deepcopy

from gpflow.models import SVGP


def build_gpr_model(data):
    variance = tf.math.reduce_variance(data.observations)
    kernel = gpflow.kernels.Matern52(variance=variance, lengthscales=[0.2, 0.2])
    model = GPR(data.astuple(), kernel, noise_variance=1e-5)
    return model


def build_svgp_model(data):
    inputs = data.query_points
    variance = tf.math.reduce_variance(data.observations)
    kernel = gpflow.kernels.Matern52(variance=variance, lengthscales=[0.2, 0.2])
    model = SVGP(kernel, gpflow.likelihoods.Gaussian(), inputs[:2], num_data=len(inputs))
    return model


def run_experiment(model_config):
    bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
    result = bo.optimize(2, initial_data, model_config)
    return result


# configuration shared by all experiments, this is modified by each experiment condition
basic_config = {"model": build_gpr_model(initial_data)}

# here we specify our experiments
experiment_conditions = [
    {"model_args": {"num_kernel_samples": 50}},
    {"model": build_svgp_model(initial_data)},
]

results = []
for exp in experiment_conditions:
    model_config = deepcopy(basic_config)
    for key in exp:
        model_config[key] = exp[key]
    results.append(run_experiment(model_config))


# %% [markdown]
# ## Registry of supported models
#
# Configuration dictionaries are made possible with the `ModelRegistry` that contains mapping between each model (e.g. GPflow or GPflux) and the corresponding model wrapper and optimizer wrapper. All models that Trieste currently supports are registered there.
#
# You can add new models to the registry, in case you have custom models with which you wish to use the configuration dictionaries. Let's see an example of this. We will register the `GPMC` model from GPflow that is currently not supported. You would likely need to create a new model wrapper and perhaps a new optimizer wrapper as well, but just for the sake of an example we will borrow here existing wrappers, `GaussianProcessRegression` and `Optimizer`.

# %%
from trieste.models import ModelRegistry

# adding the GPMC model to the registry
ModelRegistry.register_model(gpflow.models.GPMC, GaussianProcessRegression, Optimizer)

# check if it has been registered
print(gpflow.models.GPMC in ModelRegistry.get_registered_models())

# you can use the same command to get a list of all supported models
list(ModelRegistry.get_registered_models())

# %% [markdown]
# Note that you can use the same operation to overwrite an existing entry in the registry. For example, if you want to modify the interface used with a registered model and use the modified one instead.


# %% [markdown]
# ## LICENSE
#
# [Apache License 2.0](https://github.com/secondmind-labs/trieste/blob/develop/LICENSE)
