import pytest
import tensorflow as tf
import trieste
import numpy.testing as npt
from trieste.acquisition.combination import Product
from trieste.acquisition.function.entropy import MUMBO, CostWeighting
from trieste.acquisition.optimizer import generate_continuous_optimizer
from trieste.bayesian_optimizer import FrozenRecord
from trieste.data import Dataset, add_fidelity_column, check_and_extract_fidelity_query_points, get_dataset_for_fidelity
from trieste.objectives import (
    Linear2Fidelity,
    Linear3Fidelity,
    Linear5Fidelity,
    SingleObjectiveMultifidelityTestProblem,
)
from trieste.objectives.utils import mk_observer, Observer
from trieste.types import TensorType
from trieste.models.gpflow import (
    MultifidelityAutoregressive,
    build_multifidelity_autoregressive_models,
)

from tests.util.misc import random_seed

def _build_observer(problem: SingleObjectiveMultifidelityTestProblem) -> Observer:

    objective_function = problem.objective

    def noisy_objective(x: TensorType) -> TensorType:

        _, fidelities = check_and_extract_fidelity_query_points(x)
        y = objective_function(x)
        not_lowest_fidelity = fidelities > 0
        noise = tf.random.normal(y.shape, stddev=1e-1, dtype=y.dtype)
        y = tf.where(not_lowest_fidelity, y + noise, y)
        return y

    return mk_observer(noisy_objective)


def _build_nested_multifidelity_dataset(
    problem: SingleObjectiveMultifidelityTestProblem, observer: Observer
) -> Dataset:

    num_fidelities = problem.num_fidelities
    initial_sample_sizes = [10 + 2* (num_fidelities - i) for i in range(num_fidelities)]
    fidelity_samples = list()
    lowest_fidelity_sample = problem.input_search_space.sample(initial_sample_sizes[0])
    lowest_fidelity_sample = add_fidelity_column(lowest_fidelity_sample, 0)
    fidelity_samples.append(lowest_fidelity_sample)

    for i in range(1, num_fidelities):
        previous_fidelity_points = fidelity_samples[i - 1][:, :-1]
        indices = tf.range(tf.shape(previous_fidelity_points)[0])
        random_indices = tf.random.shuffle(indices)[: initial_sample_sizes[i]]
        random_points = tf.gather(previous_fidelity_points, random_indices)
        sample_points = add_fidelity_column(random_points, i)
        fidelity_samples.append(sample_points)

    query_points = tf.concat(fidelity_samples, axis=0)
    dataset = observer(query_points)

    return dataset

@random_seed
@pytest.mark.parametrize("problem", ((Linear2Fidelity), (Linear3Fidelity), (Linear5Fidelity)))
def test_multifidelity_bo_finds_minima_of_linear_problem(
    problem: SingleObjectiveMultifidelityTestProblem,
):

    observer = _build_observer(problem)
    initial_data = _build_nested_multifidelity_dataset(problem, observer)
    costs = [2.0*(n+1) for n in range(problem.num_fidelities)]
    input_search_space = problem.input_search_space  # Does not include fidelities
    search_space = problem.search_space  # Includes fidelities

    model = MultifidelityAutoregressive(
        build_multifidelity_autoregressive_models(
            initial_data,
            num_fidelities=problem.num_fidelities,
            input_search_space=input_search_space,
            kernel_priors=True,
        )
    )

    model.update(initial_data)
    model.optimize(initial_data)

    bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
    acq_builder = Product(
        MUMBO(search_space).using("OBJECTIVE"),
        CostWeighting(costs).using("OBJECTIVE"),
    )
    optimizer = generate_continuous_optimizer(num_initial_samples=10_000, num_optimization_runs=10)
    rule = trieste.acquisition.rule.EfficientGlobalOptimization(builder=acq_builder, optimizer=optimizer)

    num_steps = 5
    result = bo.optimize(num_steps, initial_data, model, acquisition_rule=rule)
    query_points_on_top = get_dataset_for_fidelity(result.try_get_final_dataset(),model.num_fidelities-1 )
    arg_min_idx = tf.squeeze(tf.argmin(query_points_on_top.observations, axis=0))
    best_x, best_y = query_points_on_top.query_points[arg_min_idx], query_points_on_top.observations[arg_min_idx]

    # check we solve the problem
    minimizer_err = tf.abs((best_x - problem.minimizers) / problem.minimizers)
    assert tf.reduce_any(tf.reduce_all(minimizer_err < 0.05, axis=-1), axis=0)
    npt.assert_allclose(best_y, problem.minimum, rtol=0.1)
