# %% [markdown]
# # Tracking and visualizing optimizations using Tensorboard

# %%
import numpy as np
import tensorflow as tf
import random

np.random.seed(1793)
tf.random.set_seed(1793)
random.seed(3)

# %% [markdown]
# We often wish to track or visualize the Bayesian optimization process, either during and following execution. This tutorial shows how to do this using the [TensorBoard](https://www.tensorflow.org/tensorboard) visualization toolkit.

# %% [markdown]
# ## Set up the problem
#
# For this tutorial, we'll use the same set up as before.

# %%
import trieste
import gpflow

search_space = trieste.space.Box([0, 0], [1, 1])
observer = trieste.objectives.utils.mk_observer(trieste.objectives.scaled_branin)
initial_query_points = search_space.sample_sobol(5)
initial_data = observer(initial_query_points)

variance = tf.math.reduce_variance(initial_data.observations)
kernel = gpflow.kernels.Matern52(variance=variance, lengthscales=[0.2, 0.2])
gpr = gpflow.models.GPR(initial_data.astuple(), kernel, noise_variance=1e-5)
gpflow.set_trainable(gpr.likelihood, False)
model = trieste.models.gpflow.GaussianProcessRegression(gpr)

# %% [markdown]
# ## Setting up TensorBoard
#
# Before running the optimization loop, we decide where TensorBoard summary logs should be stored, and create a summary writer to do this.

# %%
# Clear any logs from previous runs
# !rm -rf logs/tensorboard

summary_writer = tf.summary.create_file_writer("logs/tensorboard")

# %% [markdown]
# We can now load the TensorBoard extension, though at this point there will not be any data to dispay.

# %%
# %load_ext tensorboard
# %tensorboard --logdir "logs/tensorboard"

# %% [markdown]
# ## Running and tracking the Bayesian Optimizer
#
# By passing in the `summary_writer` to `BayesianOptimizer`, we tell trieste to log relevant information during optimization. While the optimization is running we can refresh TensorBoard to see its progress.

# %%
bo = trieste.bayesian_optimizer.BayesianOptimizer(
    observer, search_space, summary_writer=summary_writer
)

result, history = bo.optimize(15, initial_data, model).astuple()


# %% [markdown]
# ## Logging additional model parameters
#
# When logging is enabled, trieste decides what information is interesting enough to log. This includes objective and acquisition function values, and some (but not all) model parameters. To log additional model parameters, you can define your own model subclass and override the `log` method. For example, the following GPR subclass also logs the likelihood variance at each step.

# %%
class GPRExtraLogging(trieste.models.gpflow.GaussianProcessRegression):
    def log(self, summary_writer, step_number, context):
        """
        Log model-specific information at a given optimization step.

        :param summary_writer: Summary writer to log with.
        :param step_number: The current optimization step number.
        :param context: A context string to use when logging.
        """
        super().log(summary_writer, step_number, context)
        with summary_writer.as_default(step=step_number):
            tf.summary.scalar(f"{context}.likelihood.variance", self.model.likelihood.variance)


gpflow.set_trainable(gpr.likelihood, True)
model = GPRExtraLogging(gpr)

# %% [markdown]
# Running with this model now also produces logs for the variance.

# %%
bo = trieste.bayesian_optimizer.BayesianOptimizer(
    observer, search_space, summary_writer=summary_writer
)

result, history = bo.optimize(15, initial_data, model).astuple()

# %% [markdown]
# ## LICENSE
#
# [Apache License 2.0](https://github.com/secondmind-labs/trieste/blob/develop/LICENSE)
