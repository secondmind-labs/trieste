"""
Analytic version of EHVI
main reference:

@article{yang2019efficient,
  title={Efficient computation of expected hypervolume improvement using box decomposition algorithms},
  author={Yang, Kaifeng and Emmerich, Michael and Deutz, Andr{\'e} and B{\"a}ck, Thomas},
  journal={Journal of Global Optimization},
  volume={75},
  number={1},
  pages={3--34},
  year={2019},
  publisher={Springer}
}
"""
import tensorflow as tf
from itertools import product
from typing import Mapping, Union

from ...utils.pareto import Pareto
from ..function import AcquisitionFunction
from ...data import Dataset
from ...models import ProbabilisticModel
from ...type import TensorType

from tensorflow_probability import distributions as tfd
from .function import HypervolumeAcquisitionBuilder, get_nadir_point


class Expected_Hypervolume_Improvement(HypervolumeAcquisitionBuilder):
    """
        Builder for the :func:`hv_probability_of_improvement` acquisition function
        """

    def __init__(self, nadir_setting: Union[str, callable] = "default"):
        """
        :param nadir_setting the method of calculating the nadir point, either default or a callable
        """
        self._nadir_setting = nadir_setting

    def __repr__(self) -> str:
        return f"Expected Hypervolume Improvement({self._nadir_setting!r})"

    def _calculate_nadir(self, pareto: Pareto, nadir_setting="default"):
        """
        calculate the reference point for hypervolme calculation
        :param pareto: Pareto class
        :param nadir_setting
        """
        if nadir_setting == "default":
            return get_nadir_point(pareto.front)
        else:
            assert callable(nadir_setting), ValueError(
                "nadir_setting: {} do not understood".format(nadir_setting)
            )
            return nadir_setting(pareto.front)

    def prepare_acquisition_function(
            self, datasets: Mapping[str, Dataset], models: Mapping[str, ProbabilisticModel]
    ) -> AcquisitionFunction:
        """
        :param datasets: The data from the observer. Must be populated.
        :param models: The model over the specified ``dataset``. Must have event shape [1].
                       TODO: The shape is still awaiting for trieste multi-model design decision
        :return: The hv_probability_of_improvement function.
        """
        for _, data in datasets.items():
            tf.debugging.assert_positive(len(data), message='Dataset must be populated.')
        means = [models[model_tag].predict(datasets[data_tag].query_points)[0] for
                 model_tag, data_tag in zip(models, datasets)]
        for mean in means:
            tf.debugging.assert_shapes(
                [(mean, ["_", 1])], message="Expected model with event shape [1].")
        datasets_mean = tf.concat(means, axis=1)
        _PF = Pareto(Dataset(query_points=tf.zeros_like(datasets_mean), observations=datasets_mean))
        # _PF = Pareto(datasets)
        _nadir_pt = self._calculate_nadir(_PF, nadir_setting=self._nadir_setting)
        return lambda at: self._acquisition_function(models, at, _PF, _nadir_pt)

    @staticmethod
    def _acquisition_function(
            models: Mapping[str, ProbabilisticModel],
            at: TensorType,
            pareto: Pareto,
            nadir: tf.Tensor,
    ) -> tf.Tensor:
        return expected_hv_improvement(models, at, pareto, nadir)


def expected_hv_improvement(
        models: Mapping[str, ProbabilisticModel],
        at: TensorType,
        pareto: Pareto,
        nadir_point: tf.Tensor,
) -> TensorType:
    r"""
    HV calculation using Eq. 44 of original paper
    Note:
    1. Since in Trieste we do not assume the use of a certain non-dominated partition algorithm.
       we do not assume the last dimension partition has only one (lower) bound, as in the paper,
        this is not equally efficient as the original paper, but is applicable to different
        non-dominated partition algorithm
    2. The Psi and nu function in the original paper is defined for a maximization, we inverse our
       problem to make use of the same equation
    3. The calculation of EHVI based on Eq.44 is independent on the order of cells

    :param models: The model of the objective function.
    :param at: The points at which to evaluate the probability of feasibility.
                Must have rank at least two
    :param pareto: Pareto class
    :param nadir_point The reference point for calculating hypervolume
    :return: The hypervolume probability of improvement at ``at``.
    """

    normal = tfd.Normal(loc=tf.zeros(shape=1, dtype=at.dtype), scale=tf.ones(shape=1, dtype=at.dtype))

    def Psi(a, b, mean, std) -> TensorType:
        """
        Generic Expected Improvement o reference a, defined at Eq. 19 of [yang2019]
        param: a: [num_cells, out_dim] lower bounds
        param: b: [num_cells, out_dim] upper bounds
        param: mean: [..., out_dim]
        param: var: [..., out_dim]
        """
        return std * normal.prob((b - mean) / std) + \
               (mean - a) * (1 - normal.cdf((b - mean) / std))

    def nu(l, u, mean, std) -> TensorType:
        """
        Eq. 25 of [yang2019]
        Note: as we deal with minimization, we use negative version of our problem
         to make use of the original formula
        """
        return (u - l) * (1 - normal.cdf((u - mean) / std))

    predicts = [models[model_tag].predict(at) for model_tag in models]
    candidate_mean, candidate_var = (tf.concat(moment, 1) for moment in zip(*predicts))
    candidate_std = tf.sqrt(candidate_var)

    # calc ehvi assuming maximization
    neg_candidate_mean = - tf.expand_dims(candidate_mean, 1)  # [..., 1, out_dim]
    candidate_std = tf.expand_dims(candidate_std, 1)  # [..., 1, out_dim]

    lb_points, ub_points = pareto.get_partitioned_cell_bounds(nadir_point)

    neg_lb_points, neg_ub_points = - ub_points, - lb_points  # ref Note. 3

    neg_ub_points = tf.minimum(neg_ub_points, 1e10)  # this maximum: 1e10 is heuristically chosen

    psi_ub = Psi(neg_lb_points, neg_ub_points, neg_candidate_mean, candidate_std)  # [..., num_cells, out_dim]
    psi_lb = Psi(neg_lb_points, neg_lb_points, neg_candidate_mean, candidate_std)  # [..., num_cells, out_dim]

    psi_lb2ub = tf.maximum(psi_lb - psi_ub, 0.0)  # [..., num_cells, out_dim]
    nu_contrib = nu(neg_lb_points, neg_ub_points, neg_candidate_mean, candidate_std)

    # get stacked factors of Eq. 45
    # [2^m, dim_indices]
    cross_index = tf.constant(
        list(product(*[[0, 1] for _ in range(nadir_point.shape[-1])])))

    # Take the cross product of psi_diff and nu across all outcomes
    # [..., num_cells, 2(operation_num, refer Eq. 45), num_obj]
    stacked_factors = tf.concat([tf.expand_dims(psi_lb2ub, -2), tf.expand_dims(nu_contrib, -2)], axis=-2)

    # [..., num_cells, 2^m, 2(operation_num), num_obj]
    factor_combinations = tf.linalg.diag_part(tf.gather(stacked_factors, cross_index, axis=-2))

    # calculate Eq. 44
    # prod of different output_dim -> sum over 2^m combination -> sum over num_cells; [..., 1]
    return tf.reduce_sum(tf.reduce_sum(tf.reduce_prod(factor_combinations, axis=-1), axis=-1), axis=-1, keepdims=True)