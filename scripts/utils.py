from typing import *
import os
import shutil

import gpflow
import trieste
import tensorflow as tf

from trieste.acquisition.optimizer import generate_continuous_optimizer
from gpflux.layers.basis_functions.fourier_features import RandomFourierFeaturesCosine

import matplotlib.pyplot as plt


objective_kernels = {
    "eq": gpflow.kernels.RBF,
    "matern12": gpflow.kernels.Matern12,
    "matern32": gpflow.kernels.Matern32,
    "matern52": gpflow.kernels.Matern52,
}

gp_objectives = list(objective_kernels.keys())


def arg_summary(args: dict):
    return "\n".join([f"{k:<32} | {v:<64}" for k, v in args.__dict__.items()])


def make_gp_objective(
        kernel: str,
        num_fourier_components: int,
        dim: int,
        linear_stddev: float = 1e-6,
        num_initial_points: int = 10000,
        num_optimization_runs: int = 1000,
        dtype: tf.DType = tf.float64,
    ):

    assert kernel in objective_kernels

    search_space = trieste.space.Box(lower=dim*[-1.], upper=dim*[1.])

    D = search_space.dimension

    kernel = objective_kernels[kernel]()

    rff = RandomFourierFeaturesCosine(
        kernel=kernel,
        n_components=num_fourier_components,
        dtype=dtype,
    )

    weights = tf.random.normal(shape=(num_fourier_components,), dtype=dtype)
    linear_weights = linear_stddev * tf.random.normal(shape=(D,), dtype=dtype)

    objective = lambda x: tf.linalg.matvec(a=rff(x), b=weights)[:, None] + \
                          tf.reduce_sum(linear_weights[None, :] * x, axis=1)[:, None]
    reshaped_objective = lambda x: -objective(x[:, 0, :])

    continuous_optimizer = generate_continuous_optimizer(
        num_initial_samples=num_initial_points,
        num_optimization_runs=num_optimization_runs,
    )

    minimizer, _ = continuous_optimizer(
        space=search_space,
        target_func=reshaped_objective,
    )
    minimum = objective(minimizer)

    return objective, minimum, search_space


def set_up_logging(args, summary):

    # Make directory to save results
    params = f"seed-{args.search_seed}"

    if not (args.acquisition == "ei"):
        params = params + \
                 f"_batch_size-{args.batch_size}" + \
                 f"_gtol-{args.gtol}" + \
                 f"_ftol-{args.ftol}" + \
                 f"_init-designs-{args.num_initial_designs}" + \
                 f"_num-opt-{args.num_optimization_runs}"

    if "mcei" in args.acquisition:
        params = params + f"_num_mcei_samples-{args.num_mcei_samples}"

    if "qei" in args.acquisition:
        params = params + f"_num-sobol-{args.qei_sample_size}"

    if args.objective in gp_objectives:
        objective_name = f"{args.objective}_" + \
                         f"{args.objective_dimension}_" + \
                         f"{args.objective_seed}"

    else:
        objective_name = args.objective

    path = os.path.join(
        args.save_dir,
        args.acquisition,
        objective_name,
        params,
    )

    if os.path.exists(path):
        shutil.rmtree(path)

    os.makedirs(path, exist_ok=True)
    
    with open(f"{path}/arguments.txt", "w") as file:
        file.write(summary)
        file.close()

    if os.path.exists(f"{path}/log-regret.txt"):
        os.remove(f"{path}/log-regret.txt")

    if os.path.exists(f"{path}/log"):
        shutil.rmtree(f"{path}/log")

    if os.path.exists(f"{path}/figs"):
        shutil.rmtree(f"{path}/figs")

    if os.path.exists(f"{path}/time.txt"):
        os.remove(f"{path}/time.txt")
        
    os.makedirs(f"{path}/figs", exist_ok=True)

    # summary_writer = tf.summary.create_file_writer(f"{path}/log")
    # trieste.logging.set_tensorboard_writer(summary_writer)

    return path


def plot_2D_results(optimizer, path, filename):
    
    D = 2
    
    opt_results = optimizer._acquisition_rule.vectorized_child_results
    data = optimizer.model.get_internal_data()
    
    x_data = tf.convert_to_tensor(data.query_points)
    num_plots_per_side = int(len(opt_results)**0.5)+1
    
    k = 0
    plt.figure(figsize=(10, 10))
    
    for i in range(num_plots_per_side):
        for j in range(num_plots_per_side):
            
            k = k + 1
            
            x = tf.reshape(opt_results[k-1]["x"], (-1, 2))
            
            plt.subplot(num_plots_per_side, num_plots_per_side, k)
            plt.scatter(x_data[:, 0], x_data[:, 1], color="k", s=50)
            plt.scatter(x[:, 0], x[:, 1], color="red", marker="+")
            
            plt.xticks([])
            plt.yticks([])
            
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            
            if k >= len(opt_results):
                break
                
        if k >= len(opt_results):
            break
        
    plt.tight_layout()
    plt.savefig(f"{path}/figs/{filename}")
    plt.close()
    