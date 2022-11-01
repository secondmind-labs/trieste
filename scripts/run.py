import shutil
import sys
sys.path.append("./scripts")

import argparse

import tensorflow as tf
import gpflow
import numpy as np

from gpflow.utilities import print_summary

import trieste
from trieste.models.gpflow.models import GaussianProcessRegression
from trieste.ask_tell_optimization import AskTellOptimizer
from trieste.acquisition.optimizer import generate_continuous_optimizer

from trieste.acquisition.rule import (
    RandomSampling,
    EfficientGlobalOptimization,
    # EfficientGlobalOptimizationWithPreOptimization,
)

from trieste.models.gpflow.builders import (
    build_gpr,
)

from trieste.acquisition.function import (
    ExpectedImprovement,
    BatchMonteCarloExpectedImprovement,
    BatchExpectedImprovement,
    GreedyContinuousThompsonSampling,
    # DecoupledBatchMonteCarloExpectedImprovement,
    # AnnealedBatchMonteCarloExpectedImprovement,
)

from trieste.acquisition.utils import split_acquisition_function_calls

from trieste.objectives.utils import mk_observer
from trieste.objectives.single_objectives import *

from utils import (
    make_gp_objective,
    arg_summary,
    set_up_logging,
    gp_objectives,
    plot_2D_results,
)

from time import time
import pickle


standard_objectives = {
    # "scaled_branin": (
    #     scaled_branin,
    #     SCALED_BRANIN_MINIMUM,
    #     BRANIN_SEARCH_SPACE,
    # ),
    # "shekel_4": (
    #     shekel_4,
    #     SHEKEL_4_MINIMUM,
    #     SHEKEL_4_SEARCH_SPACE,
    # ),
    # "hartmann_3": (
    #     hartmann_3,
    #     HARTMANN_3_MINIMUM,
    #     HARTMANN_3_SEARCH_SPACE,
    # ),
    # "hartmann_6": (
    #     hartmann_6,
    #     HARTMANN_6_MINIMUM,
    #     HARTMANN_6_SEARCH_SPACE,
    # ),
    # "ackley_5": (
    #     ackley_5,
    #     ACKLEY_5_MINIMUM,
    #     ACKLEY_5_SEARCH_SPACE,
    # ),
}



def build_model(
        dataset: trieste.data.Dataset,
        trainable_noise: bool,
        kernel=None
    ) -> GaussianProcessRegression:
    """
    :param dataset:
    :param trainable_noise:
    :param kernel:
    :return:
    """

    variance = tf.math.reduce_variance(dataset.observations)

    if kernel is None:
        kernel = gpflow.kernels.Matern52(variance=variance)

    else:
        kernel = kernel(variance)

    gpr = gpflow.models.GPR(dataset.astuple(), kernel, noise_variance=1e-4)

    if not trainable_noise:
        gpflow.set_trainable(gpr.likelihood, False)

    return GaussianProcessRegression(gpr)


def log10_regret(queries: tf.Tensor, minimum: tf.Tensor) -> tf.Tensor:
    """
    :param queries:
    :param minimum:
    :return:
    """
    regret = tf.reshape(tf.reduce_min(queries) - minimum, shape=())
    return tf.math.log(regret) / tf.cast(tf.math.log(10.), dtype=regret.dtype)


def save_results(iteration, path, optimizer, minimum, optimisation_results=None, dt=None):

    observations = optimizer.to_result().try_get_final_dataset().observations
    log_regret = log10_regret(queries=observations, minimum=minimum)

    with open(f"{path}/log-regret.txt", "a") as file:
        file.write(f"{iteration}, {observations.shape[0]}, {log_regret}\n")
        file.close()

    with open(f"{path}/time.txt", "a") as file:
        file.write(f"{iteration}, {dt or 0.}\n")
        file.close()
        
    if optimisation_results is not None:
        with open(f"{path}/step-{iteration}.pickle", "wb") as handle:
            pickle.dump(optimisation_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return log_regret


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "acquisition",
        type=str,
        choices=[
            "random",
            "thompson",
            "ei",
            "mcei",
            "qei",
        ],
        help="Which acquisition strategy to use",
    )

    parser.add_argument(
        "objective",
        type=str,
        choices=gp_objectives,
    )

    parser.add_argument(
        "-batch_size",
        type=int,
    )

    parser.add_argument(
        "-num_batches",
        type=int,
    )

    parser.add_argument(
        "--search_seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--objective_seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--num_initial_designs",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--num_optimization_runs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--objective_dimension",
        type=int,
        default=-1,
    )

    parser.add_argument(
        "--num_gp_rff_features",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--num_obj_fourier",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--trainable_noise",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--num_mcei_samples",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--linear_stddev",
        type=float,
        default=1e-6,
    )

    parser.add_argument(
        "--qei_sample_size",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--gtol",
        type=float,
        default=1e-6,
    )

    parser.add_argument(
        "--ftol",
        type=float,
        default=1e-10,
    )

    parser.add_argument(
        "--maxiter",
        type=int,
        default=15000,
    )

    parser.add_argument(
        "--maxfun",
        type=int,
        default=15000,
    )

    parser.add_argument(
        "--freeze_gp_hypers",
        type=bool,
        default=True,
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="experiments",
    )

    # Parse arguments and print argument summary
    args = parser.parse_args()
    summary = arg_summary(args)
    print(summary)

    assert not (args.objective in gp_objectives and args.objective_dimension == -1)
    
    dtype = tf.float64

    # Set up logging: save path, argument logging, regret logging, tensorboard
    path = set_up_logging(args=args, summary=summary)

    # Set objective
    if args.objective in standard_objectives:
        objective, minimum, search_space = standard_objectives[args.objective]

    else:

        # Set objective seed
        tf.random.set_seed(args.objective_seed)
        np.random.seed(args.objective_seed)

        # Make objective
        objective, minimum, search_space = make_gp_objective(
            kernel=args.objective,
            dim=args.objective_dimension,
            num_fourier_components=args.num_obj_fourier,
            linear_stddev=args.linear_stddev,
        )

    # Set search seed
    tf.random.set_seed(args.search_seed)
    np.random.seed(args.search_seed)

    # Observe initial points
    D = int(search_space.dimension)
    num_initial_points = 2 * D + 2
    initial_query_points = search_space.sample(num_initial_points)
    observer = mk_observer(objective)
    initial_dataset = observer(initial_query_points)

    model = build_gpr(
        data=initial_dataset,
        search_space=search_space,
        likelihood_variance=1e-4,
    )

    model = GaussianProcessRegression(
        model,
        num_rff_features=args.num_gp_rff_features,
    )
    
    if args.freeze_gp_hypers:
        
        model.model.mean_function.c.assign(
            tf.convert_to_tensor(0., dtype=dtype),
        )
        
        model.model.kernel.variance.assign(
            tf.convert_to_tensor(1., dtype=dtype),
        )
        
        model.model.kernel.lengthscales.assign(
            tf.convert_to_tensor(D*[1.], dtype=dtype),
        )
        
        model.model.likelihood.variance.assign(
            tf.convert_to_tensor(1e-4, dtype=dtype),
        )
        
    print_summary(model.model)

    # Create acquisition rule
    if args.acquisition == "random":
        rule = RandomSampling(num_query_points=args.batch_size)
        num_bo_steps = args.num_batches

    else:

        if args.acquisition == "thompson":
            acquisition_function = GreedyContinuousThompsonSampling()
            num_bo_steps = args.num_batches
            batch_size = args.batch_size

        elif args.acquisition == "ei":
            acquisition_function = ExpectedImprovement()
            num_bo_steps = args.batch_size * args.num_batches
            batch_size = 1

        elif args.acquisition == "mcei":
            acquisition_function = BatchMonteCarloExpectedImprovement(
                sample_size=args.num_mcei_samples,
            )
            num_bo_steps = args.num_batches
            batch_size = args.batch_size

        elif args.acquisition == "qei":
            acquisition_function = BatchExpectedImprovement(
                sample_size=args.qei_sample_size,
                batch_size=args.batch_size,
                dtype=tf.float64,
            )
            num_bo_steps = args.num_batches
            batch_size = args.batch_size

        optimizer_args = {
            "options":
                {
                    "gtol" : args.gtol,
                    "ftol" : args.ftol,
                    "maxiter" : args.maxiter,
                    "maxfun" : args.maxfun,
                }
        }

        continuous_optimizer = generate_continuous_optimizer(
            num_initial_samples=args.num_initial_designs,
            num_optimization_runs=args.num_optimization_runs,
            optimizer_args=optimizer_args,
        )

        continuous_optimizer = split_acquisition_function_calls(
            continuous_optimizer,
            split_size=args.num_initial_designs//10,
        )

        rule = EfficientGlobalOptimization(
            num_query_points=batch_size,
            builder=acquisition_function,
            optimizer=continuous_optimizer,
        )

    # Create ask-tell optimizer
    optimizer = AskTellOptimizer(
        search_space=search_space,
        datasets=initial_dataset,
        models=model,
        acquisition_rule=rule,
        optimize_model=(not args.freeze_gp_hypers),
    )
    
    optimisation_results = None
    dt = 0.

    # Run optimization
    for i in range(num_bo_steps):
        
        log_regret = save_results(
            iteration=i,
            path=path,
            optimizer=optimizer,
            minimum=minimum,
            optimisation_results=optimisation_results,
            dt=dt,
        )
        
        print(f"Batch {i}: Log10 regret {log_regret:.3f}")
        
        start = time()
        query_batch, optimisation_results = optimizer.ask()
        dt = time() - start
        
        print_summary(model.model)
        
        query_values = observer(query_batch)
        optimizer.tell(
            query_values,
            optimize_model=(not args.freeze_gp_hypers),
        )

    log_regret = save_results(
        iteration=i+1,
        path=path,
        optimizer=optimizer,
        minimum=minimum,
        optimisation_results=optimisation_results,
        dt=dt,
    )
    
    print(f"Batch {i+1}: Log10 regret {log_regret:.3f}")


if __name__ == "__main__":
    main()
