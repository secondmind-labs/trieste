# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: 'Python 3.7.5 64-bit (''.venv'': venv)'
#     name: python3
# ---

# %% [markdown]
# # Asynchronous Bayesian optimization with Trieste
#
# In this notebook we show how Bayesian optimization can be done asynchronuosly. That is pertinent in scenarios when the objective function we are aiming to optimize can be run for several points in parallel, and observations might return back at different times. To avoid wasting resources we immediatelly request the next point asynchronuosly, taking into account points that still being evaluated.
#
# To contrast this approach with regular [batch optimization](batch_optimization.ipynb), this notebook also shows how to run parallel synchronous batch approach.

# %%
# silence TF warnings and info messages, only print errors
# https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

tf.get_logger().setLevel("ERROR")
import numpy as np
import math

import time
import timeit


# %% [markdown]
# First, let's define a simple objective that will emulate workload of variable time. We will be using class BO benchmark function [Branin](https://www.sfu.ca/~ssurjano/branin.html), and insert sleep call in the middle of the calculation to emulate delay. Our sleep delay is a scaled sum of all input values to make sure delays are uneven.
# %%
def objective(points, sleep=True):
    if points.shape[1] != 2:
        raise ValueError(f"Incorrect input shape, expected (*, 2), got {points.shape}")

    def scaled_branin(x):
        x0 = x[..., :1] * 15.0 - 5.0
        x1 = x[..., 1:] * 15.0

        b = 5.1 / (4 * math.pi ** 2)
        c = 5 / math.pi
        r = 6
        s = 10
        t = 1 / (8 * math.pi)
        scale = 1 / 51.95
        translate = -44.81

        if sleep:
            # insert some artificial delay
            # increases linearly with the absolute value of points
            # which means our evaluations will take different time
            delay = 5 * np.sum(x)
            pid = os.getpid()
            print(
                f"Process {pid}: Objective: pretends like it's doing something for {delay:.2}s",
                flush=True,
            )
            time.sleep(delay)

        return scale * ((x1 - b * x0 ** 2 + c * x0 - r) ** 2 + s * (1 - t) * np.cos(x0) + translate)

    observations = []
    for point in points:
        observation = scaled_branin(point)
        observations.append(observation)

    return np.array(observations)

# test the defined objective function
objective(np.array([[0.1, 0.5]]), sleep=False)

# %% [markdown]
# As always, we need to prepare model and some initial data to kick-start the optimization process.

# %%
from trieste.space import Box
from trieste.data import Dataset
from trieste.objectives import SCALED_BRANIN_MINIMUM

search_space = Box([0, 0], [1, 1])
num_initial_points = 3
initial_query_points = search_space.sample(num_initial_points)
initial_observations = objective(initial_query_points.numpy(), sleep=False)
initial_data = Dataset(
    query_points=initial_query_points,
    observations=tf.constant(initial_observations, dtype=tf.float64),
)

import gpflow
from trieste.models import create_model
from trieste.models.gpflow.config import GPflowModelConfig


def build_model(data):
    variance = tf.math.reduce_variance(data.observations)
    kernel = gpflow.kernels.RBF(variance=variance)
    gpr = gpflow.models.GPR(data.astuple(), kernel, noise_variance=1e-5)
    gpflow.set_trainable(gpr.likelihood, False)

    return GPflowModelConfig(
        **{
            "model": gpr,
            "optimizer": gpflow.optimizers.Scipy(),
            "optimizer_args": {
                "minimize_args": {"options": dict(maxiter=100)},
            },
        }
    )

# these imports will be used later for optimization
from trieste.acquisition import LocalPenalizationAcquisitionFunction
from trieste.acquisition.rule import AsyncEfficientGlobalOptimization, EfficientGlobalOptimization
from trieste.ask_tell_optimization import AskTellOptimizer


# %% [markdown]
# ## Multiprocessing setup
#
# To keep this notebook as reproducible as possible, we will only be using Python's multiprocessing package here. In this section we will explain our setup and define some common code to be used later.
#
# In both synchronous and asynchronous scenarios we will have a fixed set of worker processes performing observations. We will also have a main process responsible for optimization process with Trieste. When Trieste suggests a new point, it is inserted into a points queue. One of the workers picks this point from the queue, performs the observation, and inserts the output into observations queue. The main process then picks up the observation from the queue, at which moment it either waits for the rest of the points in the batch to come back (synchronous scenario) or immediately suggests a new point (asynchrunous scenario). This process continues either for a certain number of iterations or until we accumulate necessary number of observations.
# 
# The overall setup is illustrated in this diagram:
# ![multiprocessing setup](figures/async_bo.png)

# %%
# Necessary multiprocessing primitives
from multiprocessing import Manager, Process

# %% [markdown]
# We now define several common functions to implement the described setup. First we define a worker function that will be running a single observation in a separate process. Worker takes both queues as an input, reads next point from the points queue, makes an observation, and inserts observed data into the observations queue.

# %%

def observer_proc(points_queue, observations_queue):
    pid = os.getpid()

    while True:
        point_to_observe = points_queue.get()
        if point_to_observe is None:
            return

        print(f"Process {pid}: Observer : observing data at point {point_to_observe}", flush=True)
        new_observation = objective(point_to_observe, sleep=enable_sleep_delays)
        new_data = (point_to_observe, new_observation)

        print(f"Process {pid}: Observer : observed data {new_data}", flush=True)

        observations_queue.put(new_data)

# %% [markdown]
# Next we define two helper functions, one is to create a certain number of worker processes, and another is to terminate them once we are done.

# %%

def create_worker_processes(n_workers, points_queue, obseverations_queue):
    observer_processes = []
    for i in range(n_workers):
        worker_proc = Process(target=observer_proc, args=(points_queue, obseverations_queue))
        worker_proc.daemon = True
        worker_proc.start()

        observer_processes.append(worker_proc)

    return observer_processes

def terminate_processes(processes):
    for prc in processes:
        prc.terminate()
        prc.join()
        prc.close()

# %% [markdown]
# Finally we set some common parameters. See comments below for explanation of what each one means.
# %%
# Number of worker processes to run simultaneously
# Setting this to 1 will turn both setups into non-batch sequential optimization
num_workers = 3
# Number of iterations to run the sycnhronous scenario for
num_iterations = 10
# Number of observations to collect in asynchronous scenario
num_observations = num_workers * num_iterations
# Set this flag to False to disable sleep delays in case you want the notebook to execute quickly
enable_sleep_delays = True

# %% [markdown]
# ## Asynchronous optimization
# This section runs the asynchronous optimization routine. We first setup the [ask/tell optimizer](ask_tell_optimization.ipynb). Notice that we use **AsyncEfficientGlobalOptimization** rule specifically designed for asynchronous scenarios. Next we create thread-safe queues for points and observations, and run the optimization loop.
#
# Crucially, even though we are using batch acquisition function Local Penalization, we specify batch size of 1. This is because we don't really want a batch. Since the amout of workers we have is fixed, whenever we see a new observation we only need one point back. However this process can only be done with acquisition funcitons that implement greedy batch collection strategies, because they are able to take into account points that are currently being observed (in Trieste we call them "pending"). Trieste currently provides two such functions: Local Penalization and GIBBON.

# %%

# setup Ask Tell BO
model_spec = build_model(initial_data)
model = create_model(model_spec)

local_penalization_acq = LocalPenalizationAcquisitionFunction(search_space, num_samples=2000)
local_penalization_rule = AsyncEfficientGlobalOptimization(num_query_points=1, builder=local_penalization_acq)  # type: ignore

async_bo = AskTellOptimizer(search_space, initial_data, model, local_penalization_rule)

# retrieve process id for nice logging
pid = os.getpid()
# create point and observation queues
m = Manager()
pq = m.Queue()
oq = m.Queue()
# keep track of all workers we have launched
observer_processes = []
# counter to keep track of collected observations
points_observed = 0

start = timeit.default_timer()
try:
    observer_processes = create_worker_processes(num_workers, pq, oq)

    # init the queue with first batch of points
    for _ in range(num_workers):
        point = async_bo.ask()
        pq.put(np.atleast_2d(point.numpy()))

    while points_observed < num_observations:
        # keep asking queue for new observations until one arrives
        try:
            new_data = oq.get_nowait()
            print(f"Process {pid}: Main     : received data {new_data}", flush=True)
        except:
            continue

        # new_data is a tuple of (point, observation value)
        # here we turn it into a Dataset and tell of it Trieste
        points_observed += 1
        new_data = Dataset(
            query_points=tf.constant(new_data[0], dtype=tf.float64),
            observations=tf.constant(new_data[1], dtype=tf.float64),
        )
        async_bo.tell(new_data)

        # now we can ask Trieste for one more point
        # and feed that back into the points queue
        point = async_bo.ask()
        print(f"Process {pid}: Main     : acquired point {point}", flush=True)
        pq.put(np.atleast_2d(point))
finally:
    terminate_processes(observer_processes)
stop = timeit.default_timer()

# Collect the observations, compute the running time
async_lp_observations = async_bo.to_result().try_get_final_dataset().observations - SCALED_BRANIN_MINIMUM
async_lp_time = stop - start
print(f"Got {len(async_lp_observations)} observations in {async_lp_time:.2f}s")

# %% [markdown]
# ## Synchronous parallel optimization
#
# This section runs the synchronous parallel optimization with Trieste. We again use Local Penalization acquisition function, but this time with batch size equal to the number of workers we have available. Once Trieste suggests the batch, we add all points to the point queue, and workers immediatelly pick them up, one point per worker. Therefore all points in the batch are evaluated in parallel.

# %%
# setup Ask Tell BO
model_spec = build_model(initial_data)
model = create_model(model_spec)

local_penalization_acq = LocalPenalizationAcquisitionFunction(search_space, num_samples=2000)
local_penalization_rule = EfficientGlobalOptimization(  # type: ignore
    num_query_points=num_workers, builder=local_penalization_acq
)

sync_bo = AskTellOptimizer(search_space, initial_data, model, local_penalization_rule)


# retrieve process id for nice logging
pid = os.getpid()
# create point and observation queues
m = Manager()
pq = m.Queue()
oq = m.Queue()
# keep track of all workers we have launched
observer_processes = []

start = timeit.default_timer()
try:
    observer_processes = create_worker_processes(num_workers, pq, oq)

    # BO loop starts here
    for i in range(num_iterations):
        print(f"Process {pid}: Main     : iteration {i} starts", flush=True)

        # get a batch of points from Trieste, send them to points queue
        # each worker picks up a point and processes it
        points = sync_bo.ask()
        for point in points.numpy():
            pq.put(point.reshape(1, -1))  # reshape is to make point a 2d array

        # now we wait for all workers to finish
        # we create an empty dataset and wait
        # until we collected as many observations in it
        # as there were points in the batch
        all_new_data = Dataset(
            tf.zeros((0, initial_data.query_points.shape[1]), tf.float64),
            tf.zeros((0, initial_data.observations.shape[1]), tf.float64),
        )
        while len(all_new_data) < num_workers:
            # this line blocks the process until new data is available in the queue
            new_data = oq.get()
            print(f"Process {pid}: Main     : received data {new_data}", flush=True)

            new_data = Dataset(
                query_points=tf.constant(new_data[0], dtype=tf.float64),
                observations=tf.constant(new_data[1], dtype=tf.float64),
            )

            all_new_data = all_new_data + new_data

        # tell Trieste of new batch of observations
        sync_bo.tell(all_new_data)

finally:
    terminate_processes(observer_processes)
stop = timeit.default_timer()

# Collect the observations, compute the running time
sync_lp_observations = (
    sync_bo.to_result().try_get_final_dataset().observations - SCALED_BRANIN_MINIMUM
)
sync_lp_time = stop - start
print(f"Got {len(sync_lp_observations)} observations in {sync_lp_time:.2f}s")


# %% [markdown]
# ## Comparison
# To compare outcomes of sync and async runs, let's plot their respective regrets side by side, and print out the running time. We expect async scenario to run a little bit faster for this toy problem.

# %%
from trieste.objectives import SCALED_BRANIN_MINIMUM

from util.plotting import plot_regret
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2)

sync_lp_min_idx = tf.squeeze(tf.argmin(sync_lp_observations, axis=0))
async_lp_min_idx = tf.squeeze(tf.argmin(async_lp_observations, axis=0))

plot_regret(
    sync_lp_observations.numpy(), ax[0], num_init=len(initial_data), idx_best=sync_lp_min_idx
)
ax[0].set_yscale("log")
ax[0].set_ylabel("Regret")
ax[0].set_ylim(0.0000001, 100)
ax[0].set_xlabel("# evaluations")
ax[0].set_title(f"Sync LP, {len(sync_lp_observations)} points, time {sync_lp_time:.2f}")

plot_regret(
    async_lp_observations.numpy(), ax[1], num_init=len(initial_data), idx_best=async_lp_min_idx
)
ax[1].set_yscale("log")
ax[1].set_ylabel("Regret")
ax[1].set_ylim(0.0000001, 100)
ax[1].set_xlabel("# evaluations")
ax[1].set_title(f"Async LP, {len(async_lp_observations)} points, time {async_lp_time:.2f}s")

fig.tight_layout()

# %%
