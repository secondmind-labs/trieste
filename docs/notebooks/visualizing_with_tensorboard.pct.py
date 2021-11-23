# %% [markdown]
# # Tracking and visualizing optimizations using Tensorboard

# %%
import numpy as np
import tensorflow as tf

np.random.seed(1793)
tf.random.set_seed(1793)

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
# Before running the optimization loop, we decide where TensorBoard summary logs should be stored, and set a summary writer to do this.

# %%
# Clear any logs from previous runs
# !rm -rf logs/tensorboard

summary_writer = tf.summary.create_file_writer("logs/tensorboard/experiment1")
trieste.logging.set_tensorboard_writer(summary_writer)

# %% [markdown]
# We can now also load the TensorBoard extension, though at this point there will not be any data to dispay.

# %% magic_args="echo Loading TensorBoard..." language="script"
#
# %load_ext tensorboard
# %tensorboard --logdir "logs/tensorboard"

# %% [markdown]
# ![TensorBoard empty page](figures/tensorboard_empty.png)

# %% [markdown]
# ## Running and tracking the Bayesian Optimizer
#
# By setting the summary writer, we tell Trieste to log relevant information during optimization. While the optimization is running, we can refresh TensorBoard to see its progress.

# %%
num_steps = 15
bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
result, history = bo.optimize(num_steps, initial_data, model).astuple()


# %% [markdown]
# The data remains available when the optimization is over.
#
# ![TensorBoard graphs](figures/tensorboard_bo.png)

# %% [markdown]
# ## Logging additional model parameters
#
# When logging is enabled, Trieste decides what information is interesting enough to log. This includes objective and acquisition function values, and some (but not all) model parameters. To log additional model parameters, you can define your own model subclass and override the `log` method. For example, the following GPR subclass also logs the average lengthscale at each step.

# %%
class GPRExtraLogging(trieste.models.gpflow.GaussianProcessRegression):

    def log(self):
        super().log()
        summary_writer = trieste.logging.get_tensorboard_writer()
        if summary_writer:
            with summary_writer.as_default(step=trieste.logging.get_step_number()):
                tf.summary.scalar(
                    "kernel.lengthscales.mean",
                    np.mean(self.get_kernel().lengthscales)
                )


model = GPRExtraLogging(gpr)

# %% [markdown]
# Running with this model now also produces logs for the mean lengthscale. We mark this optimization run as a separate experiment by creating a new summary writer. Tensorboard will conveniently offer an automatic comparison of the experiments.

# %%
summary_writer = tf.summary.create_file_writer("logs/tensorboard/experiment2")
trieste.logging.set_tensorboard_writer(summary_writer)

bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
result, history = bo.optimize(num_steps, initial_data, model).astuple()

# %% [markdown]
# ![TensorBoard custom graphs](figures/tensorboard_custom.png)

# %% [markdown]
# ## Logging additional acqusition rule metrics
#
#
# Similarly, it is possible to log additional metrics connected to the acquisition rule by overriding rule's `acquire` method (or any other method used while evaluating the rule). For example, the following class also logs the mean coordinates of the selected points:

# %%
class EGOExtraLogging(trieste.acquisition.rule.EfficientGlobalOptimization):
    
    def acquire(self, search_space, models, datasets = None):
        points = super().acquire(search_space, models, datasets)
        summary_writer = trieste.logging.get_tensorboard_writer()
        if summary_writer:
            with summary_writer.as_default(step=trieste.logging.get_step_number()):
                tf.summary.scalar("EGO.points_selected.mean", tf.math.reduce_mean(points))
        return points
    
summary_writer = tf.summary.create_file_writer("logs/tensorboard/experiment3")
trieste.logging.set_tensorboard_writer(summary_writer)

bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
result, history = bo.optimize(  # type: ignore
    num_steps, initial_data, model, acquisition_rule=EGOExtraLogging()
).astuple()

# %% [markdown]
# ![TensorBoard custom rule graphs](figures/tensorboard_custom_rule.png)

# %% [markdown]
# ## Using Tensorboard with Ask-Tell Optimization
#
# To use Tensorboard logging with the [Ask-Tell interface](ask_tell_optimization.ipynb), you must also explicitly set the optimization step number before calling ask or tell:

# %%
summary_writer = tf.summary.create_file_writer("logs/tensorboard/experiment4")
trieste.logging.set_tensorboard_writer(summary_writer)

ask_tell = trieste.ask_tell_optimization.AskTellOptimizer(search_space, initial_data, model)

for step in range(num_steps):
    trieste.logging.set_step_number(step)
    new_point = ask_tell.ask()
    new_data = observer(new_point)
    ask_tell.tell(new_data)

# %% [markdown]
# ## LICENSE
#
# [Apache License 2.0](https://github.com/secondmind-labs/trieste/blob/develop/LICENSE)
