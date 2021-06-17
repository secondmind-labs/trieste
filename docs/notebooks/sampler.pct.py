# %% [markdown]
# # Custom sampler for Initial Design of Experiment

# %%
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# %%
import trieste

np.random.seed(1793)
tf.random.set_seed(1793)
# %% [markdown]
# ## Describe the problem
# In this example, we look into the initial Design of Experiment sampler using Random, Halton sequence and Sobol Sequence
#
#

# %% [markdown]
# Sample 50 initial points Randomly
# %%
search_space = trieste.space.Box([0., 0.], [1., 1.])
num_initial_points = 100
initial_query_points = search_space.sample(num_initial_points)
plt.scatter(initial_query_points[:, 0], initial_query_points[:, 1])

# %% [markdown]
# Sample 50 initial points using Halton sequence
# %%
search_space = trieste.space.Box([0., 0.], [1., 1.])
num_initial_points = 100
initial_query_points = search_space.sample_halton(num_initial_points)
plt.scatter(initial_query_points[:, 0], initial_query_points[:, 1])

# %% [markdown]
# Sample 50 initial points using Sobol sequence
# %%
search_space = trieste.space.Box([0., 0.], [1., 1.])
num_initial_points = 100
initial_query_points = search_space.sample_sobol(num_initial_points)
plt.scatter(initial_query_points[:, 0], initial_query_points[:, 1])
