# %% [markdown]
# # Ask-Tell Optimization Interface

# %% [markdown]
# In this notebook we will illustrate the use of Ask-Tell interface in Trieste. This is a useful interface to have in the cases when you want to have greater control of the optimization loop, or if letting Trieste manage this loop is impossible.
#
# First, some commmon code we will be using in the notebook.

# %%
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gpflow

from trieste.ask_tell_optimization import AskTellOptimizer
from trieste.bayesian_optimizer import OptimizationResult, Record
from trieste.data import Dataset
from trieste.models.gpflow import GPflowModelConfig
from trieste.objectives import scaled_branin, SCALED_BRANIN_MINIMUM
from trieste.objectives.utils import mk_observer
from trieste.observer import OBJECTIVE
from trieste.space import Box

from util.plotting import plot_regret

np.random.seed(1234)
tf.random.set_seed(1234)


search_space = Box([0, 0], [1, 1])
n_steps = 5

def build_model(data, kernel_func=None):
    """kernel_func should be a function that takes variance as a single input parameter"""
    variance = tf.math.reduce_variance(data.observations)
    if kernel_func is None:
        kernel = gpflow.kernels.Matern52(variance=variance)
    else:
        kernel = kernel_func(variance)
    gpr = gpflow.models.GPR(data.astuple(), kernel, noise_variance=1e-5)
    gpflow.set_trainable(gpr.likelihood, False)

    return GPflowModelConfig(**{
        "model": gpr,
        "optimizer": gpflow.optimizers.Scipy(),
    })

num_initial_points = 5
initial_query_points = search_space.sample(num_initial_points)
observer = mk_observer(scaled_branin)
initial_data = observer(initial_query_points)
model = build_model(initial_data)

# %% [markdown]
# ## Timing acquisition function: simple use case for Ask-Tell
#
# Let's say we are very concerned with the performance of the acqusition function, and want a simple way of measuring its performance over the course of the optimization. At the time of writing these lines, regular Trieste's optimizer does not provide such coustomization functionality, and this is where Ask-Tell comes in handy.

# %%
import timeit

ask_tell = AskTellOptimizer(search_space, initial_data, model)

for step in range(n_steps):
    start = timeit.default_timer()
    new_point = ask_tell.ask()
    stop = timeit.default_timer()

    print(f"Time at step {step + 1}: {stop - start}")

    new_data = observer(new_point)
    ask_tell.tell(new_data)


# %% [markdown]
# Once ask-tell optimization is over, you can get an optimization result object from it and perform whatever analysis you need. Just like with regular Trieste optimization interface. For instance here we will plot for each optimization step

# %%
def plot_ask_tell_regret(ask_tell_result):
    observations = ask_tell_result.try_get_final_dataset().observations.numpy()
    arg_min_idx = tf.squeeze(tf.argmin(observations, axis=0))

    suboptimality = observations - SCALED_BRANIN_MINIMUM.numpy()
    _, ax = plt.subplots(1, 2)
    plot_regret(suboptimality, ax[0], num_init=num_initial_points, idx_best=arg_min_idx)

    ax[0].set_yscale("log")
    ax[0].set_ylabel("Regret")
    ax[0].set_ylim(0.001, 100)
    ax[0].set_xlabel("# evaluations")

plot_ask_tell_regret(ask_tell.to_result())

# %% [markdown]
# ## Model switching: using only Ask part
#
# A slightly more complex use case. Let's suppose we want to switch between two models depending on some criteria dynamically during the optimization loop. Why would we do it? [Who knows!](https://bayesopt.github.io/papers/2014/paper13.pdf). In that case we can only use Ask part of the Ask-Tell interface.

# %%
model1 = build_model(initial_data, kernel_func=lambda v: gpflow.kernels.RBF(variance=v))
model2 = build_model(initial_data, kernel_func=lambda v: gpflow.kernels.Matern32(variance=v))

dataset = initial_data
for step in range(n_steps):
    # this criterion is meaningless
    # but hopefully illustrates the idea!
    if step % 2 == 0:
        print("Using model 1")
        model = model1
    else:
        print("Using model 2")
        model = model2

    ask_tell = AskTellOptimizer(search_space, dataset, model)
    new_point = ask_tell.ask()

    new_data_point = observer(new_point)

    dataset = dataset + new_data_point

plot_ask_tell_regret(ask_tell.to_result())


# %% [markdown]
# ## External experiment: manipulating optimization state
#
# Now let's suppose you are optimizing a real life process, e.g. a lab experiment. This time you cannot even express the objective function in Python code. Instead you would like to ask Trieste what configuration to run next, go to the lab, perform the experiment, collect data, feed it back to Trieste and ask for the next configuration, and so on. In such case you want to save the optimization state and then load it later.
#
# Of course we cannot perform a real physical experiment within this notebook, so we will just mimick it.

# %%
for step in range(n_steps):
    print(f"Ask Trieste for configuration #{step}")
    new_config = ask_tell.ask()

    print("Saving Trieste state to re-use later")
    state: Record[None] = ask_tell.to_record()

    print(f"In the lab running the experiment #{step}.")
    new_datapoint = scaled_branin(new_config)
    print("Back from the lab")

    print("Restore optimizer from the saved state")
    ask_tell = AskTellOptimizer.from_record(state, search_space)

    print("Tell optimizer the new datapoint")
    ask_tell.tell(Dataset(new_config, new_datapoint))


plot_ask_tell_regret(ask_tell.to_result())


# %% [markdown]
# A word of warning. This may not work with writing optimization state to disk, and reading it back later. So use with caution!