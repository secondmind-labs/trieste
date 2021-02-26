from abc import ABC, abstractmethod
from typing import Mapping
import tensorflow as tf
from ...utils.pareto import Pareto
from ..function import BatchAcquisitionFunctionBuilder, Dataset, \
    ProbabilisticModel, AcquisitionFunction, AcquisitionFunctionBuilder


# TODO: This is still awaiting the multi-model design decision from trieste
class MultiModelBatchAcquisitionBuilder(ABC):
    """
    Convenience acquisition function builder for a batch acquisition function (or component of a
    composite batch acquisition function) that requires only one model, dataset pair.
    """

    def using(self, tags: [str]) -> BatchAcquisitionFunctionBuilder:
        """
        :param tags: NOT IN USE ATM
        :return: A batch acquisition function builder that selects the model and dataset specified
            by ``tag``, as defined in :meth:`prepare_acquisition_function`.
        """
        multi_builder = self

        class _Anon(BatchAcquisitionFunctionBuilder):
            def prepare_acquisition_function(
                    self, datasets: Mapping[str, Dataset], models: Mapping[str, ProbabilisticModel]
            ) -> AcquisitionFunction:
                # TODO: seems can create subdicts instead of using the original one
                return multi_builder.prepare_acquisition_function(datasets, models)

            def __repr__(self) -> str:
                return f"{multi_builder!r} using tag {tags!r}"

        return _Anon()

    @abstractmethod
    def prepare_acquisition_function(
            self, dataset: Mapping[str, Dataset], model: Mapping[str, ProbabilisticModel]
    ) -> AcquisitionFunction:
        """
        :param dataset: The data to use to build the acquisition function.
        :param model: The model over the specified ``dataset``.
        :return: An acquisition function.
        """


class MultiModelAcquisitionBuilder(ABC):
    """
        Convenience acquisition function builder for a batch acquisition function (or component of a
        composite batch acquisition function) that requires only one model, dataset pair.
        """

    def using(self, tags: [str]) -> AcquisitionFunctionBuilder:
        """
        :param tags: NOT IN USE ATM
        :return: A batch acquisition function builder that selects the model and dataset specified
            by ``tag``, as defined in :meth:`prepare_acquisition_function`.
        """
        multi_builder = self

        class _Anon(BatchAcquisitionFunctionBuilder):
            def prepare_acquisition_function(
                    self, datasets: Mapping[str, Dataset], models: Mapping[str, ProbabilisticModel]
            ) -> AcquisitionFunction:
                # TODO: seems can create subdicts instead of using the original one
                return multi_builder.prepare_acquisition_function(datasets, models)

            def __repr__(self) -> str:
                return f"{multi_builder!r} using tag {tags!r}"

        return _Anon()

    @abstractmethod
    def prepare_acquisition_function(
            self, dataset: Mapping[str, Dataset], model: Mapping[str, ProbabilisticModel]
    ) -> AcquisitionFunction:
        """
        :param dataset: The data to use to build the acquisition function.
        :param model: The model over the specified ``dataset``.
        :return: An acquisition function.
        """


class HypervolumeAcquisitionBuilder(MultiModelAcquisitionBuilder):
    @abstractmethod
    def _calculate_nadir(self, pareto: Pareto, nadir_setting="default"):
        """
        calculate the reference point for hypervolme calculation
        :param pareto: Pareto class
        :param nadir_setting
        """

        
class HypervolumeBatchAcquisitionBuilder(MultiModelBatchAcquisitionBuilder):
    @abstractmethod
    def _calculate_nadir(self, pareto: Pareto, nadir_setting="default"):
        """
        calculate the reference point for hypervolme calculation
        :param pareto: Pareto class
        :param nadir_setting
        """


def get_nadir_point(front: tf.Tensor) -> tf.Tensor:
    """
    nadir point calculation method
    """
    f = tf.math.reduce_max(front, axis=0, keepdims=True) - tf.math.reduce_min(
        front, axis=0, keepdims=True
    )
    return tf.math.reduce_max(front, axis=0, keepdims=True) + 2 * f / front.shape[0]
