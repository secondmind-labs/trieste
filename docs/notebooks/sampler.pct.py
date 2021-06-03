# %% [markdown]
# # Custom sampler for Initial Design of Experiment

# %%
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# %%
import trieste
from trieste.utils import design

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
doe = design.Random()
search_space = trieste.space.Box([0., 0.], [1., 1.], doe)
num_initial_points = 50
initial_query_points = search_space.sample(num_initial_points)
plt.scatter(initial_query_points[:, 0], initial_query_points[:, 1])

# %% [markdown]
# Sample 50 initial points using Halton sequence
# %%
doe = design.HaltonSequence()
search_space = trieste.space.Box([0., 0.], [1., 1.], doe)
num_initial_points = 50
initial_query_points = search_space.sample(num_initial_points)
plt.scatter(initial_query_points[:, 0], initial_query_points[:, 1])

# %% [markdown]
# Sample 50 initial points using Sobol sequence
# %%
doe = design.SobolSequence()
search_space = trieste.space.Box([0., 0.], [1., 1.], doe)
num_initial_points = 50
initial_query_points = search_space.sample(num_initial_points)
plt.scatter(initial_query_points[:, 0], initial_query_points[:, 1])
