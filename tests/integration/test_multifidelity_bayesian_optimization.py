import pytest
import tensorflow as tf
import trieste
from trieste.acquisition.combination import Product
from trieste.acquisition.function.entropy import MUMBO, CostWeighting
from trieste.acquisition.optimizer import generate_continuous_optimizer
from trieste.bayesian_optimizer import FrozenRecord
from trieste.data import Dataset, add_fidelity_column, check_and_extract_fidelity_query_points
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

# def noisy_linear_2_fidelity(x: TensorType) -> TensorType:

#     _, fidelities = check_and_extract_fidelity_query_points(x)
#     y = linear_two_fidelity(x)
#     not_lowest_fidelity = fidelities > 0
#     noise = tf.random.normal(y.shape, stddev=1e-1, dtype=y.dtype)
#     y = tf.where(not_lowest_fidelity, y + noise, y)
#     return y


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
    initial_sample_sizes = [2 + 2 * (num_fidelities - i) for i in range(num_fidelities)]
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
    print(dataset)

    return dataset


@pytest.mark.parametrize("problem", ((Linear2Fidelity), (Linear3Fidelity), (Linear5Fidelity)))
def test_multifidelity_bo_finds_minima_of_linear_problem(
    problem: SingleObjectiveMultifidelityTestProblem,
):

    observer = _build_observer(problem)
    initial_data = _build_nested_multifidelity_dataset(problem, observer)
    low_cost = 1.0
    high_cost = 4.0
    input_search_space = problem.input_search_space  # Does not include fidelities
    search_space = problem.search_space  # Includes fidelities

    model = MultifidelityAutoregressive(
        build_multifidelity_autoregressive_models(
            initial_data,
            num_fidelities=problem.num_fidelities,
            input_search_space=input_search_space,
        )
    )

    model.update(initial_data)
    model.optimize(initial_data)

    bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)
    acq_builder = Product(
        MUMBO(search_space).using("OBJECTIVE"),
        CostWeighting(low_cost, high_cost).using("OBJECTIVE"),
    )
    optimizer = generate_continuous_optimizer(num_initial_samples=10_000, num_optimization_runs=10)
    rule = trieste.acquisition.rule.EfficientGlobalOptimization(builder=acq_builder)

    num_steps = 1
    result = bo.optimize(num_steps, initial_data, model, acquisition_rule=rule)

    dataset = result.try_get_final_dataset()
