import tensorflow as tf
from abc import abstractmethod
from ...utils.pareto import Pareto
from ...type import TensorType
from ..function import SingleModelAcquisitionBuilder


class HypervolumeAcquisitionBuilder(SingleModelAcquisitionBuilder):
    @abstractmethod
    def _calculate_ref_pt(self, pareto: Pareto, ref_pt_calc_method="default"):
        """
        calculate the reference point for hypervolme calculation
        :param pareto: Pareto class
        :param ref_pt_calc_method
        """


def get_reference_point(front: TensorType) -> TensorType:
    """
    reference point calculation method
    """
    f = tf.math.reduce_max(front, axis=0) - tf.math.reduce_min(front, axis=0)
    return tf.math.reduce_max(front, axis=0) + 2 * f / front.shape[0]
