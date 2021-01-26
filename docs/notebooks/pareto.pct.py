# %%
from dataclasses import astuple
import trieste
import gpflow
from gpflow.utilities import set_trainable
from gpflow import default_float
import numpy as np
import tensorflow as tf
from trieste.type import TensorType
from trieste.utils.pareto import Pareto
from trieste.acquisition.function import AcquisitionFunctionBuilder, AcquisitionFunction
from trieste.type import QueryPoints
from typing import Mapping
import tensorflow_probability as tfp
from trieste.utils.pareto import non_dominated_sort
from trieste.data import Dataset
from matplotlib import pyplot as plt

# %%
__stability__ = 1e-6
gpflow.config.set_default_float(np.float64)
np.random.seed(1793)
tf.random.set_seed(1793)

def vlmop2(x: TensorType) -> TensorType:
    transl = 1 / np.sqrt(2)
    part1 = (x[:, 0] - transl) ** 2 + (x[:, 1] - transl) ** 2
    part2 = (x[:, 0] + transl) ** 2 + (x[:, 1] + transl) ** 2
    y1 = 1 - tf.exp(-1 * part1)
    y2 = 1 - tf.exp(-1 * part2)
    return tf.stack([y1, y2], axis=1)

mins = [-2, -2]
maxs = [ 2,  2]

lower_bound = tf.cast(mins, gpflow.default_float())
upper_bound = tf.cast(maxs, gpflow.default_float())
search_space = trieste.space.Box(lower_bound, upper_bound)

num_initial_points = 200
num_objective = 2
initial_query_points = search_space.sample(num_initial_points)

OBJECTIVE1 = "OBJECTIVE1"
OBJECTIVE2 = "OBJECTIVE2"

def observer(query_points):
    """
    return Dataset: has 2 attributes: query_points(means x), observations
    """
    y = vlmop2(query_points)
    print('query_points: {}'.format(query_points))
    print('Get: {}'.format(y))
    return {
        OBJECTIVE1: trieste.data.Dataset(query_points, y[:, 0, np.newaxis]),
        OBJECTIVE2: trieste.data.Dataset(query_points, y[:, 1, np.newaxis])
    }

#%%
initial_data = observer(initial_query_points)
print(type(initial_data["OBJECTIVE1"]))
front, dom = non_dominated_sort(initial_data)

# %%
plt.scatter(initial_data["OBJECTIVE1"].observations,initial_data["OBJECTIVE2"].observations)
plt.scatter(front[:,0],front[:,1])
