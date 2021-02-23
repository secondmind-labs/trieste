"""
Monte Carlo & Reparameteriztaion trick based EHVI
"""

from typing import Mapping
from math import inf
import tensorflow as tf

from trieste.acquisition import AcquisitionFunctionBuilder, AcquisitionFunction, SingleModelAcquisitionBuilder
from trieste.data import Dataset
from trieste.models import ProbabilisticModel
from trieste.utils.pareto import Pareto
from trieste.type import TensorType


class MonteCarloHypervolumeExpectedImprovement(AcquisitionFunctionBuilder):
    def __init__(self, num_samples: int = 512):
        super().__init__()
        self.num_samples = num_samples

    # TODO: Maybe use of state in the future
    def prepare_acquisition_function(
            self, datasets: Mapping[str, Dataset], models: Mapping[str, ProbabilisticModel]
    ) -> AcquisitionFunction:
        """
        :param dataset: The data from the observer.
        :param model: The model over the specified ``dataset``.
        :return: The expected improvement function.
        """
        eps_shape = [self.num_samples, len(models)]
        eps = tf.random.normal(eps_shape, dtype=tf.float64)  # [S, L]

        datasets_mean = tf.concat([models[model_tag].predict(datasets[data_tag].query_points)[0]
                                   for model_tag, data_tag in zip(models, datasets)], axis=1)
        # datasets_mean, _ = (tf.concat(moment, 1) for moment in zip(*predicts))
        _Pareto = Pareto(Dataset(query_points=tf.zeros_like(datasets_mean), observations=datasets_mean))
        nadir = get_nadir_point(_Pareto.front)
        return lambda at: self._acquisition_function(models, at, _Pareto, nadir, eps)

    @staticmethod
    def _acquisition_function(
            models: Mapping[str, ProbabilisticModel],
            at: TensorType,
            pareto: Pareto,
            nadir: tf.Tensor,
            eps
    ) -> tf.Tensor:
        return MC_HVEI(models, at, pareto, nadir, eps)


def predict_independent_f_samples_with_reparametrisation_trick(
        model: ProbabilisticModel, at: TensorType, eps: tf.Tensor
) -> tf.Tensor:
    """
    Returns independent samples according to the reparametrization trick.
    The sample size S is determined by eps,
    N is the number of query points and L is the number of latent processes.
    :param model:
    :param at: N query points [N, D]
    :param eps: a [S, L] tensor of N(0, 1) i.i.d. samples
    :return: a [S, N, L] tensor
    """
    assert len(at.shape) == 2
    mean, cov = model.predict(at)  # both [N, L]
    return mean[None, ...] + tf.math.sqrt(cov)[None, ...] * eps[:, None, :]  # [S, N, L]


# TODO: Write Test
def MC_HVEI(
        models: Mapping[str, ProbabilisticModel],
        at: TensorType,
        pareto: Pareto,
        nadir_point: tf.Tensor,
        eps
) -> tf.Tensor:
    r"""
    :param models: The model of the objective function.
    :param at: The points at which to evaluate the probability of feasibility.
                Must have rank at least two
    :param pareto: Pareto class
    :param nadir_point The reference point for calculating hypervolume
    :return: The hypervolume probability of improvement at ``at``.
    """
    predict_samples = tf.concat([predict_independent_f_samples_with_reparametrisation_trick(
        list(models.values())[index], at, eps[:, index, tf.newaxis]) for index in range(len(models))], axis=-1)
    # [f_samples, batch_size, out_dim]

    # TODO: Include a mc_samples dimension: [f_samples, B, L]
    num_samples = tf.shape(predict_samples)[0]
    N = tf.shape(predict_samples)[1]
    outdim = tf.shape(predict_samples)[2]

    num_cells = tf.shape(pareto.bounds.lb)[0]
    # pf_ext_size * outdim
    pf_ext = tf.concat(
        [
            -inf * tf.ones([1, outdim], dtype=pareto.front.dtype),
            pareto.front,
            nadir_point,
        ],
        0,
    )
    # calculation of HV
    col_idx = tf.tile(tf.range(outdim), (num_cells,))
    ub_idx = tf.stack((tf.reshape(pareto.bounds.ub, [-1]), col_idx), axis=1)  # 上界索引
    lb_idx = tf.stack((tf.reshape(pareto.bounds.lb, [-1]), col_idx), axis=1)  # 下界索引

    ub_points = tf.reshape(tf.gather_nd(pf_ext, ub_idx), [num_cells, outdim])
    lb_points = tf.reshape(tf.gather_nd(pf_ext, lb_idx), [num_cells, outdim])
    # TODO: 这里是核心, reduce_all 是为了确保每个维度都是大于predict sample的
    # splus_valid: 返回的是 每个Batch sample是否支配该cell，因此shape为(num_cells, B)
    splus_valid = tf.reduce_all(
        tf.tile(ub_points[tf.newaxis, :, tf.newaxis, :], [num_samples, 1, N, 1]) > tf.expand_dims(predict_samples,
                                                                                                  axis=1), axis=-1)  # num_cells x B
    # splus_valid = tf.transpose(splus_valid, perm=[1, 0, 2])
    splus_idx = tf.expand_dims(tf.cast(splus_valid, dtype=ub_points.dtype), -1)
    # splus_lb = tf.tile(tf.expand_dims(lb_points, 1), [1, N, 1])
    splus_lb = tf.tile(lb_points[tf.newaxis, :, tf.newaxis, :], [num_samples, 1, N, 1])
    # 这里是最核心的地方: 替换
    splus_lb = tf.maximum(splus_lb, tf.expand_dims(predict_samples, 1))
    splus_ub = tf.tile(ub_points[tf.newaxis, :, tf.newaxis, :], [num_samples, 1, N, 1])  # 上界维持不变
    splus = tf.concat([splus_idx, splus_ub - splus_lb], axis=-1)
    # splus = tf.transpose(splus, perm=[1, 0, 2, 3])

    # Hv = tf.transpose(tf.reduce_sum(tf.reduce_prod(splus, axis=2), axis=0, keepdims=True)) #
    Hv = tf.transpose(tf.reduce_sum(tf.reduce_prod(splus, axis=-1), axis=1, keepdims=True))  #
    tf.debugging.check_numerics(
        Hv, 'NAN in HV: {}'.format(Hv), name=None
    )
    # try:
    #     tf.debugging.check_numerics(
    #         Hv, 'NAN in HV: {}'.format(Hv), name=None
    #     )
    # except:
    #     # FIXME: debug usage
    #     predicts = [models[model_tag].predict(at) for model_tag in models]
    #     candidate_mean, candidate_var = (tf.concat(moment, 1) for moment in zip(*predicts))
    #
    #     predicts2 = [models[model_tag].model.predict_f_debug(at) for model_tag in models]
    #     candidate_mean2, candidate_var2 = (tf.concat(moment, 1) for moment in zip(*predicts2))
    #
    #     predicts3 = [models[model_tag].model.predict_f(at) for model_tag in models]
    #     candidate_mean3, candidate_var3 = (tf.concat(moment, 1) for moment in zip(*predicts3))
    #     a = 2
    return tf.reduce_mean(Hv, axis=-1)
    # approach1 = tf.reduce_mean(Hv, axis=-1)
    # approach1_hv = Hv

    # for i in range(num_samples):
    #     N = tf.shape(predict_samples[0])[0]
    #     outdim = tf.shape(predict_samples[0])[1]
    #     num_cells = tf.shape(pareto.bounds.lb)[0]
    #     candidate_mean = predict_samples[i]
    #     # pf_ext_size * outdim
    #     pf_ext = tf.concat(
    #         [
    #             -inf * tf.ones([1, outdim], dtype=candidate_mean.dtype),
    #             pareto.front,
    #             nadir_point,
    #         ],
    #         0,
    #     )


#
#     col_idx = tf.tile(tf.range(outdim), (num_cells,))
#     ub_idx = tf.stack((tf.reshape(pareto.bounds.ub, [-1]), col_idx), axis=1)
#     lb_idx = tf.stack((tf.reshape(pareto.bounds.lb, [-1]), col_idx), axis=1)
#     ub_points = tf.reshape(tf.gather_nd(pf_ext, ub_idx), [num_cells, outdim])
#     lb_points = tf.reshape(tf.gather_nd(pf_ext, lb_idx), [num_cells, outdim])
#     splus_valid = tf.reduce_all(
#         tf.tile(tf.expand_dims(ub_points, 1), [1, N, 1]) > candidate_mean, axis=2
#     )  # num_cells x N
#     splus_idx = tf.expand_dims(tf.cast(splus_valid, dtype=ub_points.dtype), -1)
#     splus_lb = tf.tile(tf.expand_dims(lb_points, 1), [1, N, 1])
#     splus_lb = tf.maximum(splus_lb, candidate_mean)
#     splus_ub = tf.tile(tf.expand_dims(ub_points, 1), [1, N, 1])
#     splus = tf.concat([splus_idx, splus_ub - splus_lb], axis=2)
#     Hv = tf.transpose(tf.reduce_sum(tf.reduce_prod(splus, axis=2), axis=0, keepdims=True))
#     approach0 = Hv
#     print('Difference at iter: {}: {}'.format(i, approach1_hv[:, :, i] - approach0))


# -----------------------------------------------------------------------------------------------
# 现在我们来试试只用reshape的
# predict_samples = tf.transpose(tf.concat(predict_samples, axis=-1),
#                                perm=[1, 0, 2])  # haven't check transpose, [batch_size, f_samples, dim]
# # reshape for format consistent
# predict_samples = tf.reshape(predict_samples, shape=(num_samples * tf.shape(at)[0], len(models)))
#
# N = tf.shape(predict_samples)[0]
# outdim = tf.shape(predict_samples)[1]
# num_cells = tf.shape(pareto.bounds.lb)[0]
# # pf_ext_size * outdim
# pf_ext = tf.concat(
#     [
#         -inf * tf.ones([1, outdim], dtype=pareto.front.dtype),
#         pareto.front,
#         nadir_point,
#     ],
#     0,
# )
# # calculation of HV
# col_idx = tf.tile(tf.range(outdim), (num_cells,))
# ub_idx = tf.stack((tf.reshape(pareto.bounds.ub, [-1]), col_idx), axis=1)
# lb_idx = tf.stack((tf.reshape(pareto.bounds.lb, [-1]), col_idx), axis=1)
#
# ub_points = tf.reshape(tf.gather_nd(pf_ext, ub_idx), [num_cells, outdim])
# lb_points = tf.reshape(tf.gather_nd(pf_ext, lb_idx), [num_cells, outdim])
# splus_valid = tf.reduce_all(
#     tf.tile(tf.expand_dims(ub_points, 1), [1, N, 1]) > predict_samples, axis=2
# )  # num_cells x N
# splus_idx = tf.expand_dims(tf.cast(splus_valid, dtype=ub_points.dtype), -1)
# splus_lb = tf.tile(tf.expand_dims(lb_points, 1), [1, N, 1])
# splus_lb = tf.maximum(splus_lb, predict_samples)
# splus_ub = tf.tile(tf.expand_dims(ub_points, 1), [1, N, 1])
# splus = tf.concat([splus_idx, splus_ub - splus_lb], axis=2)
# Hv = tf.transpose(tf.reduce_sum(tf.reduce_prod(splus, axis=2), axis=0, keepdims=True))
# Hv = tf.reshape(Hv, shape=(num_samples, tf.shape(at)[0], 1))
# approach2 = tf.reduce_mean(Hv, axis=0)
#
# print(approach1 - approach2)
# a = 1


def get_nadir_point(front: tf.Tensor) -> tf.Tensor:
    """
    nadir point calculation method
    """
    f = tf.math.reduce_max(front, axis=0, keepdims=True) - tf.math.reduce_min(
        front, axis=0, keepdims=True
    )
    return tf.math.reduce_max(front, axis=0, keepdims=True) + 2 * f / front.shape[0]
