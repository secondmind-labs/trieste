import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
from trieste.objectives import (
    build_dgp_prior_function,
    DGP_MICH_2_MINIMUM,
    DGP_MICH_2_SEARCH_SPACE,
    DGP_MICH_5_MINIMUM,
    DGP_MICH_5_SEARCH_SPACE,
    DGP_ACKLEY_2_MINIMUM,
    DGP_ACKLEY_2_SEARCH_SPACE,
    DGP_ACKLEY_5_MINIMUM,
    DGP_ACKLEY_5_SEARCH_SPACE,
    michalewicz_5,
    MICHALEWICZ_5_MINIMUM,
    MICHALEWICZ_5_SEARCH_SPACE,
    ackley_5,
    ACKLEY_5_MINIMUM,
    ACKLEY_5_SEARCH_SPACE,
    branin,
    BRANIN_MINIMUM,
    BRANIN_SEARCH_SPACE,
    NOISY_HARTMANN_6_SEARCH_SPACE,
    NOISY_HARTMANN_6_MINIMUM,
    NOISY_ACKLEY_5_SEARCH_SPACE,
    NOISY_ACKLEY_5_MINIMUM,
    NOISY_SHEKEL_SEARCH_SPACE,
    NOISY_SHEKEL_MINIMUM,
    NOISY_MICH_5_SEARCH_SPACE,
    NOISY_MICH_5_MINIMUM,
    NOISY_MICH_10_SEARCH_SPACE,
    NOISY_MICH_10_MINIMUM,
    noisy_hartmann_6,
    noisy_ackley_5,
    noisy_mich_5,
    noisy_mich_10,
    noisy_shekel_4
)

from exp_utils import (
    build_vanilla_dgp_model,
    build_svgp_model,
    build_gp_model,
    build_sgpr_model,
    normalize
)

from trieste.objectives.utils import mk_observer
from trieste.ask_tell_optimization import AskTellOptimizer
import trieste
from trieste.data import Dataset
from trieste.space import SearchSpace
import pandas as pd
import time
import argparse

tf.keras.backend.set_floatx("float64")

parser = argparse.ArgumentParser()
parser.add_argument('output_filename', type=str, help='output filename', nargs='?', default='test')
parser.add_argument('--exp_name', type=str, help='experiment name', nargs='?', default='test')
parser.add_argument('--function', type=str, help='objective function', nargs='?', default='dgpmich2')
parser.add_argument('--model', type=str, help='model name', nargs='?', default='svgp')
parser.add_argument('--lnt', dest='ln', help='whether to learn noise variance', action='store_true')
parser.add_argument('--lnf', dest='ln', help='whether to learn noise variance', action='store_false')
parser.add_argument('--rtt', dest='rt', help='whether to retrain', action='store_true')
parser.add_argument('--rtf', dest='rt', help='whether to retrain', action='store_false')
parser.add_argument('--rt_every', type=int, help='how often to retrain', nargs='?', default=5)
parser.add_argument('--normt', dest='norm', help='whether to normalize data', action='store_true')
parser.add_argument('--normf', dest='norm', help='whether to normalize data', action='store_false')
parser.add_argument('--epochs', type=int, help='number of gradient steps', nargs='?', default=2000)
parser.add_argument('--num_query', type=int, help='batch size of acquistion', nargs='?', default=1)
parser.add_argument('--num_inducing', type=int, help='number of inducing points (per layer)', nargs='?', default=100)
parser.add_argument('--fix_ips_t', dest='fix_ips', help='whether to fix inducing points', action='store_true')
parser.add_argument('--fix_ips_f', dest='fix_ips', help='whether to fix inducing points', action='store_false')
parser.add_argument('--run', type=int, help='run number', nargs='?', default=1)
args = parser.parse_args()

function_key = args.function
model_key = args.model
learn_noise = args.ln
retrain = args.rt
norm = args.norm
num_inducing = args.num_inducing
retrain_every = args.rt_every
epochs = args.epochs
fix_ips = args.fix_ips
num_query = args.num_query
run = args.run

np.random.seed(run)
tf.random.set_seed(run)

function_dict = {
    "branin": [branin, BRANIN_MINIMUM, BRANIN_SEARCH_SPACE, 5, 15],
    "michalewicz5": [michalewicz_5, MICHALEWICZ_5_MINIMUM, MICHALEWICZ_5_SEARCH_SPACE, 20, 180],
    "ackley": [ackley_5, ACKLEY_5_MINIMUM, ACKLEY_5_SEARCH_SPACE, 20, 180],
    "dgpmich2": [build_dgp_prior_function('mich_2'), DGP_MICH_2_MINIMUM, DGP_MICH_2_SEARCH_SPACE,
                 10, 40],
    "dgpmich5": [build_dgp_prior_function('mich_5'), DGP_MICH_5_MINIMUM, DGP_MICH_5_SEARCH_SPACE,
                 20, 180],
    "dgpack2": [build_dgp_prior_function('ackley_2'), DGP_ACKLEY_2_MINIMUM,
                DGP_ACKLEY_2_SEARCH_SPACE, 10, 40],
    "dgpack5": [build_dgp_prior_function('ackley_5'), DGP_ACKLEY_5_MINIMUM,
                DGP_ACKLEY_5_SEARCH_SPACE, 20, 180],
    "noisymich5": [noisy_mich_5, NOISY_MICH_5_MINIMUM, NOISY_MICH_5_SEARCH_SPACE, 100, 49],
    "noisyackley5": [noisy_ackley_5, NOISY_ACKLEY_5_MINIMUM, NOISY_ACKLEY_5_SEARCH_SPACE, 100, 49],
    "noisyshekel": [noisy_shekel_4, NOISY_SHEKEL_MINIMUM, NOISY_SHEKEL_SEARCH_SPACE, 100, 49],
    "noisyhart6": [noisy_hartmann_6, NOISY_HARTMANN_6_MINIMUM, NOISY_HARTMANN_6_SEARCH_SPACE, 100,
                   49],
    "noisymich10": [noisy_mich_10, NOISY_MICH_10_MINIMUM, NOISY_MICH_10_SEARCH_SPACE, 100, 49]
}

model_dict = {
    "deepgp": [build_vanilla_dgp_model],
    "svgp": [build_svgp_model],
    "gp": [build_gp_model],
    "sgpr": [build_sgpr_model]
}

if not os.path.exists(os.path.join('results_{}'.format(args.exp_name), function_key)):
    os.makedirs(os.path.join('results_{}'.format(args.exp_name), function_key))

pd.DataFrame({
    'function': [function_key],
    'model': [model_key],
    'learn_noise': [learn_noise],
    'retrain': [retrain],
    'rt_every': [args.rt_every],
    'norm': [norm],
    'epochs': [epochs],
    'run': [run]
}).to_csv(args.output_filename)

function = function_dict[function_key][0]
F_MINIMUM = function_dict[function_key][1]

search_space = function_dict[function_key][2]
observer = mk_observer(function)

bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

num_initial_points = function_dict[function_key][3]
num_acquisitions = function_dict[function_key][4]

if retrain:
    num_loops = num_acquisitions // retrain_every


def run_bayes_opt(
    model_key: str,
    initial_data: Dataset,
    search_space: SearchSpace
) -> None:
    start_time = time.time()

    builder = model_dict[model_key][0]

    # Run Bayes Opt
    if norm:
        y_sta, y_mean, y_std = normalize(initial_data.observations)
    else:
        y_sta, y_mean, y_std = initial_data.observations, 0., 1.

    dataset = initial_data

    normalized_dataset = Dataset(initial_data.query_points, y_sta)

    for step in range(num_acquisitions):
        if retrain and step % retrain_every == 0:
            model, acquisition_rule = builder(normalized_dataset, learn_noise=learn_noise,
                                              search_space=search_space, epochs=epochs,
                                              num_inducing=num_inducing, num_query_points=num_query,
                                              fix_ips=fix_ips)
            model.optimize(normalized_dataset)
        elif step == 0:
            model, acquisition_rule = builder(normalized_dataset, learn_noise=learn_noise,
                                              search_space=search_space, epochs=epochs,
                                              num_inducing=num_inducing, num_query_points=num_query,
                                              fix_ips=fix_ips)
            model.optimize(normalized_dataset)
        else:
            model.update(normalized_dataset)
            model.optimize(normalized_dataset)

        # Ask for a new point to observe
        ask_tell = AskTellOptimizer(search_space, normalized_dataset, model, acquisition_rule,
                                    fit_model=False)
        query_point = ask_tell.ask()

        new_data_point = observer(query_point)
        dataset = dataset + new_data_point

        y_sta, _, _ = normalize(dataset.observations, y_mean, y_std)
        normalized_dataset = Dataset(dataset.query_points, y_sta)

        result_arg_min_idx = tf.squeeze(tf.argmin(dataset.observations.numpy(), axis=0))

        print(f"observation "
              f"{function_key} {run}: {dataset.observations.numpy()[result_arg_min_idx, :]}")

    # Get results
    result_dataset = dataset

    result_query_points = result_dataset.query_points.numpy()
    result_observations = result_dataset.observations.numpy()

    result_arg_min_idx = tf.squeeze(tf.argmin(result_observations, axis=0))

    pd.DataFrame(result_query_points).to_csv(
        'results_{}/{}/{}_ln{}_rt{}_{}_norm{}_nq{}_ni{}_query_points_{}'.format(args.exp_name, function_key,
                                                            model_key, learn_noise, retrain,
                                                            retrain_every, norm, num_query,
                                                            num_inducing, run))
    pd.DataFrame(result_observations).to_csv(
        'results_{}/{}/{}_ln{}_rt{}_{}_norm{}_nq{}_ni{}_observations_{}'.format(args.exp_name, function_key,
                                                            model_key, learn_noise, retrain,
                                                            retrain_every, norm, num_query,
                                                            num_inducing, run))

    print(f"{model_key} ln {learn_noise} rt {retrain} observation "
          f"{function_key} {run}: {result_observations[result_arg_min_idx, :]}")
    print("Time: ", time.time() - start_time)


initial_query_points = search_space.sample_sobol(num_initial_points)
initial_data = observer(initial_query_points)

if model_key == 'rs':
    acquired_query_points = search_space.sample(num_acquisitions*num_query)
    acquired_data = observer(acquired_query_points)

    result_query_points = tf.concat([initial_data.query_points, acquired_data.query_points], 0).numpy()
    result_observations = tf.concat([initial_data.observations, acquired_data.observations], 0).numpy()

    pd.DataFrame(result_query_points).to_csv(
        'results_{}/{}/{}_query_points_{}'.format(args.exp_name, function_key, model_key, run))
    pd.DataFrame(result_observations).to_csv(
        'results_{}/{}/{}_observations_{}'.format(args.exp_name, function_key, model_key, run))

    quit()

run_bayes_opt(model_key, initial_data, search_space)
