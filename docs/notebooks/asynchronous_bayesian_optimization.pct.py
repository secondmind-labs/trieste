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
# ### Simple objective: branin with sleeps to emulate delay. It has sleep delays to simulate workers returning at different times.

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

enable_sleep_delays = True


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
            # which means our evaluations will take different time, good for exploring async
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


# %%
objective(np.array([[0.1, 0.5]]), sleep=False)

# %% [markdown]
# ## Here comes Trieste
#
# First, define some initial data

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

initial_data

# %% [markdown]
# Model definition

# %%
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


# %% [markdown]
# Define a worker that runs a single observation in a separate process. Worker reads next point from the points queue, makes an observation, and inserts observed data into the observations queue.

# %%
import psutil


def observer_proc(points_queue, observations_queue, cpu_id):
    pid = os.getpid()

    current_process = psutil.Process()
    current_process.cpu_affinity([cpu_id])
    print(f"Process {pid}: set CPU to {cpu_id}", flush=True)

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
# Set how many workers can run simultaneously, and iterations for sync case

# %%
num_workers = 3
num_iterations = 10

# %% [markdown]
# ## Normal sync BO with LP
#
# We use LP acquisition function, and run all batch points in parallel. For that we setup the worker processes, as many as there are

# %%
from trieste.acquisition import LocalPenalizationAcquisitionFunction
from trieste.acquisition.rule import AsyncEGO, EfficientGlobalOptimization
from trieste.ask_tell_optimization import AskTellOptimizer
from multiprocessing import Manager, Process

# setup Ask Tell BO
model_spec = build_model(initial_data)
model = create_model(model_spec)

local_penalization_acq = LocalPenalizationAcquisitionFunction(search_space, num_samples=2000)
local_penalization_rule = EfficientGlobalOptimization(
    num_query_points=num_workers, builder=local_penalization_acq
)

sync_bo = AskTellOptimizer(search_space, initial_data, model, local_penalization_rule)


# thread-safe queues
m = Manager()
points_q = m.Queue()
observations_q = m.Queue()
observer_processes = []
pid = os.getpid()

start = timeit.default_timer()
try:
    # setup workers
    for i in range(psutil.cpu_count())[:num_workers]:
        observer_p = Process(target=observer_proc, args=(points_q, observations_q, i))
        observer_p.daemon = True
        observer_p.start()

        observer_processes.append(observer_p)

    # BO loop starts here
    for i in range(num_iterations):
        print(f"Process {pid}: Main     : iteration {i} starts", flush=True)

        # get a batch of points, send them to queue
        # each worker picks up a point and processes it
        points = sync_bo.ask()
        for point in points.numpy():
            points_q.put(point.reshape(1, -1))

        # this is sync scenario
        # so now we wait for all workers to finish
        # we assume no failures here
        all_new_data = Dataset(
            tf.zeros((0, initial_data.query_points.shape[1]), tf.float64),
            tf.zeros((0, initial_data.observations.shape[1]), tf.float64),
        )
        while len(all_new_data) < num_workers:
            # this line blocks the process until new data is available in the queue
            new_data = observations_q.get()
            print(f"Process {pid}: Main     : received data {new_data}", flush=True)

            new_data = Dataset(
                query_points=tf.constant(new_data[0], dtype=tf.float64),
                observations=tf.constant(new_data[1], dtype=tf.float64),
            )

            all_new_data = all_new_data + new_data

        sync_bo.tell(all_new_data)

finally:
    # cleanup workers
    for prc in observer_processes:
        prc.terminate()
        prc.join()
        prc.close()
stop = timeit.default_timer()

sync_lp_observations = (
    sync_bo.to_result().try_get_final_dataset().observations - SCALED_BRANIN_MINIMUM
)

sync_lp_time = stop - start
print(f"Got {len(sync_lp_observations)} in {sync_lp_time:.2f}s")

# %% [markdown]
# ## Async BO

# %%
from multiprocessing import Process, Manager

# setup Ask Tell BO
model_spec = build_model(initial_data)
model = create_model(model_spec)

local_penalization_acq = LocalPenalizationAcquisitionFunction(search_space, num_samples=2000)
local_penalization_rule = AsyncEGO(num_query_points=1, builder=local_penalization_acq)

bo = AskTellOptimizer(search_space, initial_data, model, local_penalization_rule)


m = Manager()
pq = m.Queue()
oq = m.Queue()
observer_processes = []
pid = os.getpid()

points_observed = 0
start = timeit.default_timer()
try:
    for i in range(psutil.cpu_count())[:num_workers]:
        observer_p = Process(target=observer_proc, args=(pq, oq, i))
        observer_p.daemon = True
        observer_p.start()

        observer_processes.append(observer_p)

    # init the queue with first batch of points
    for _ in range(num_workers):
        point = bo.ask()
        pq.put(np.atleast_2d(point.numpy()))

    while points_observed < len(sync_lp_observations) - len(initial_data):
        try:
            new_data = oq.get_nowait()
            print(f"Process {pid}: Main     : received data {new_data}", flush=True)
        except:
            continue

        new_data = Dataset(
            query_points=tf.constant(new_data[0], dtype=tf.float64),
            observations=tf.constant(new_data[1], dtype=tf.float64),
        )

        bo.tell(new_data)
        point = bo.ask()
        print(f"Process {pid}: Main     : acquired point {point}", flush=True)
        pq.put(np.atleast_2d(point))
        points_observed += 1
finally:
    for prc in observer_processes:
        prc.terminate()
        prc.join()
        prc.close()
stop = timeit.default_timer()

async_lp_observations = bo.to_result().try_get_final_dataset().observations - SCALED_BRANIN_MINIMUM

async_lp_time = stop - start
print(f"Got {len(async_lp_observations)} in {async_lp_time:.2f}s")

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
