import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
from trieste.objectives import (
    michalewicz_5,
    MICHALEWICZ_5_MINIMUM,
    MICHALEWICZ_5_SEARCH_SPACE,
    ackley_5,
    ACKLEY_5_MINIMUM,
    ACKLEY_5_SEARCH_SPACE,
    branin,
    BRANIN_MINIMUM,
    BRANIN_SEARCH_SPACE
)

from exp_utils import (
    build_vanilla_dgp_model,
    build_gp_model
)

from trieste.objectives.utils import mk_observer
import trieste
from trieste.data import Dataset
from trieste.space import SearchSpace
from trieste.acquisition.rule import DiscreteThompsonSampling
import pandas as pd
import time
import argparse
import gpflow

tf.keras.backend.set_floatx("float64")

parser = argparse.ArgumentParser()
parser.add_argument('output_filename', type=str, help='output filename', nargs='?', default='test')
parser.add_argument('--function', type=str, help='objective function', nargs='?', default='ackley')
parser.add_argument('--model', type=str, help='model name', nargs='?', default='deepgp')
parser.add_argument('--lnt', dest='ln', help='whether to learn noise variance', action='store_true')
parser.add_argument('--lnf', dest='ln', help='whether to learn noise variance', action='store_false')
parser.add_argument('--rtt', dest='rt', help='whether to retrain', action='store_true')
parser.add_argument('--rtf', dest='rt', help='whether to retrain', action='store_false')
parser.add_argument('--rt_freq', type=int, help='how often to retrain', nargs='?', default=10)
parser.add_argument('--normt', dest='norm', help='whether to normalize data', action='store_true')
parser.add_argument('--normf', dest='norm', help='whether to normalize data', action='store_false')
parser.add_argument('--run', type=int, help='run number', nargs='?', default=0)
args = parser.parse_args()

function_key = args.function
model_key = args.model
learn_noise = True
retrain = True
norm = args.norm
run = args.run

np.random.seed(run)
tf.random.set_seed(run)

function_dict = {
    "branin": [branin, BRANIN_MINIMUM, BRANIN_SEARCH_SPACE],
    "michalewicz5": [michalewicz_5, MICHALEWICZ_5_MINIMUM, MICHALEWICZ_5_SEARCH_SPACE],
    "ackley": [ackley_5, ACKLEY_5_MINIMUM, ACKLEY_5_SEARCH_SPACE],
}

model_dict = {
    "deepgp": [build_vanilla_dgp_model],
    "gp": [build_gp_model]
}

if not os.path.exists(os.path.join('results', function_key)):
    os.makedirs(os.path.join('results', function_key))

pd.DataFrame({
    'function': [function_key],
    'model': [model_key],
    'learn_noise': [learn_noise],
    'retrain': [retrain],
    'run': [run]
}).to_csv(args.output_filename)

function = function_dict[function_key][0]
F_MINIMIZER = function_dict[function_key][1]

search_space = function_dict[function_key][2]
observer = mk_observer(function)

bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

num_initial_points = 20
num_acquisitions = 480


def run_bayes_opt(
    model_key: str,
    initial_data: Dataset,
    search_space: SearchSpace
) -> None:
    start_time = time.time()

    builder = model_dict[model_key][0]

    # Run Bayes Opt
    if retrain:
        num_acq_per_loop = args.rt_freq
        num_loops = num_acquisitions // num_acq_per_loop

        current_dataset = initial_data

        for loop in range(num_loops):
            model, acquisition_rule = builder(current_dataset, learn_noise=learn_noise,
                                              search_space=search_space)
            result = bo.optimize(num_acq_per_loop, current_dataset, model,
                                 acquisition_rule=acquisition_rule, track_state=False)
            current_dataset = result.try_get_final_dataset()

            result_arg_min_idx = tf.squeeze(tf.argmin(current_dataset.observations.numpy(), axis=0))

            print(f"observation "
                  f"{function_key} {run}: {current_dataset.observations.numpy()[result_arg_min_idx, :]}")
    else:
        model, acquisition_rule = builder(initial_data, learn_noise=learn_noise, search_space=search_space)

        result = bo.optimize(num_acquisitions, initial_data, model,
                             acquisition_rule=acquisition_rule, track_state=False)

    # Get results
    result_dataset = result.try_get_final_dataset()

    result_query_points = result_dataset.query_points.numpy()
    result_observations = result_dataset.observations.numpy()

    result_arg_min_idx = tf.squeeze(tf.argmin(result_observations, axis=0))

    pd.DataFrame(result_query_points).to_csv(
        'results_cts/{}/{}_ln{}_rt{}_query_points_{}'.format(function_key, model_key, learn_noise, retrain, run))
    pd.DataFrame(result_observations).to_csv(
        'results_cts/{}/{}_ln{}_rt{}_observations_{}'.format(function_key, model_key, learn_noise, retrain, run))

    print(f"{model_key} ln {learn_noise} rt {retrain} observation "
          f"{function_key} {run}: {result_observations[result_arg_min_idx, :]}")
    print("Time: ", time.time() - start_time)


initial_query_points = search_space.sample_sobol(num_initial_points)
initial_data = observer(initial_query_points)

if model_key == 'rs':
    acquired_query_points = search_space.sample(num_acquisitions)
    acquired_data = observer(acquired_query_points)

    result_query_points = tf.concat([initial_data.query_points, acquired_data.query_points], 0).numpy()
    result_observations = tf.concat([initial_data.observations, acquired_data.observations], 0).numpy()

    pd.DataFrame(result_query_points).to_csv(
        'results_cts/{}/{}_query_points_{}'.format(function_key, model_key, run))
    pd.DataFrame(result_observations).to_csv(
        'results_cts/{}/{}_observations_{}'.format(function_key, model_key, run))

    quit()

run_bayes_opt(model_key, initial_data, search_space)
