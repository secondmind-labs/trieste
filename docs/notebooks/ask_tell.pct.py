# %% [markdown]
# # Ask tell interface

# %%
import numpy as np
import tensorflow as tf

from trieste.acquisition.rule import OBJECTIVE, EfficientGlobalOptimization
from trieste.bayesian_optimizer import optimize
from trieste.models import GaussianProcessRegression

np.random.seed(1793)
tf.random.set_seed(1793)

# %% [markdown]
# ##

# %%
import trieste
from trieste.utils.objectives import branin, mk_observer

search_space = trieste.space.Box([0, 0], [1, 1])
observer = mk_observer(branin, OBJECTIVE)
initial_data = observer(search_space.sample(5))

# %% [markdown]
# ##

# %%
import gpflow

variance = tf.math.reduce_variance(initial_data[OBJECTIVE].observations)
kernel = gpflow.kernels.Matern52(variance, [0.2, 0.2])
gpr = gpflow.models.GPR(initial_data[OBJECTIVE].astuple(), kernel, noise_variance=1e-5)
gpflow.set_trainable(gpr.likelihood, False)
models = {OBJECTIVE: GaussianProcessRegression(gpr)}

# %% [markdown]
# ##

# %%
ask_tell = optimize(search_space, initial_data, models, EfficientGlobalOptimization())
new_point = next(ask_tell)

if new_point.is_ok:
    new_data = observer(new_point.unwrap())
    new_point = ask_tell.send(new_data)

# todo I can't really progress without knowing exactly what an ask-tell optimizer needs to do
