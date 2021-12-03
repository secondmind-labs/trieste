import tensorflow as tf
from typing import cast, Mapping, Tuple

from trieste.acquisition.function import AcquisitionFunction, ExpectedImprovement, PenalizationFunction, expected_improvement, ExpectedConstrainedImprovement, GreedyAcquisitionFunctionBuilder, soft_local_penalizer, UpdatablePenalizationFunction
from trieste.data import Dataset
from trieste.types import TensorType
from trieste.space import SearchSpace
from trieste.observer import OBJECTIVE


class HackedExpectedImprovement(ExpectedImprovement):
    def update_acquisition_function(
        self, function, model, dataset=None
    ):

        tf.debugging.Assert(dataset is not None, [])
        dataset = cast(Dataset, dataset)
        tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")
        tf.debugging.Assert(isinstance(function, expected_improvement), [])
        mean, _ = model.predict(dataset.query_points)
        eta = tf.reduce_min(mean, axis=0)
        # function.update(eta)  # type: ignore
        # return function
        return expected_improvement(model, eta)


class HackedExpectedConstrainedImprovement(ExpectedConstrainedImprovement):
    def update_acquisition_function(
        self, function, models, datasets=None):
        """
        :param function: The acquisition function to update.
        :param models: The models for each tag.
        :param datasets: The data from the observer.
        """
        tf.debugging.Assert(datasets is not None, [])
        datasets = cast(Mapping[str, Dataset], datasets)
        objective_model = models[self._objective_tag]
        objective_dataset = datasets[self._objective_tag]
        tf.debugging.assert_positive(
            len(objective_dataset),
            message="Expected improvement is defined with respect to existing points in the"
            " objective data, but the objective data is empty.",
        )
        tf.debugging.Assert(self._constraint_fn is not None, [])

        self._constraint_fn = self._constraint_builder.prepare_acquisition_function(
            models, datasets=datasets
        )
        pof = self._constraint_fn(objective_dataset.query_points[:, None, ...])
        is_feasible = tf.squeeze(pof >= self._min_feasibility_probability, axis=-1)

        if not tf.reduce_any(is_feasible):
            return self._constraint_fn

        feasible_query_points = tf.boolean_mask(objective_dataset.query_points, is_feasible)
        feasible_mean, _ = objective_model.predict(feasible_query_points)
        self._update_expected_improvement_fn(objective_model, feasible_mean)

        # if self._constrained_improvement_fn is not None:
        #     return self._constrained_improvement_fn

        # @tf.function
        def constrained_function(x: TensorType) -> TensorType:
            return cast(AcquisitionFunction, self._expected_improvement_fn)(x) * cast(
                AcquisitionFunction, self._constraint_fn
            )(x)
        self._constrained_improvement_fn = constrained_function
        return self._constrained_improvement_fn


    def _update_expected_improvement_fn(
        self, objective_model, feasible_mean
    ) -> None:
        eta = tf.reduce_min(feasible_mean, axis=0)

        # if self._expected_improvement_fn is None:
        #     self._expected_improvement_fn = expected_improvement(objective_model, eta)
        # else:
        #     tf.debugging.Assert(isinstance(self._expected_improvement_fn, expected_improvement), [])
        #     self._expected_improvement_fn.update(eta)  # type: ignore
        self._expected_improvement_fn = expected_improvement(objective_model, eta)



class HackedLocalPenalizationAcquisitionFunction(GreedyAcquisitionFunctionBuilder):
    def __init__(
        self,
        search_space: SearchSpace,
        num_samples: int = 500,
        penalizer = None,
        base_acquisition_function_builder = None,
    ):
        tf.debugging.assert_positive(num_samples)
        self._search_space = search_space
        self._num_samples = num_samples
        self._lipschitz_penalizer = soft_local_penalizer if penalizer is None else penalizer
        if base_acquisition_function_builder is None:
            self._base_builder = ExpectedImprovement()
        else:
            self._base_builder = base_acquisition_function_builder
        self._lipschitz_constant = None
        self._eta = None
        self._base_acquisition_function = None
        self._penalization = None
        self._penalized_acquisition = None

    def prepare_acquisition_function(
        self,
        models,
        datasets = None,
        pending_points = None,
    ) -> AcquisitionFunction:
        """
        :param model: The model.
        :param dataset: The data from the observer. Must be populated.
        :param pending_points: The points we penalize with respect to.
        :return: The (log) expected improvement penalized with respect to the pending points.
        :raise tf.errors.InvalidArgumentError: If the ``dataset`` is empty.
        """
        # tf.debugging.Assert(dataset is not None, [])
        # dataset = cast(Dataset, dataset)
        # tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")

        acq = self._update_base_acquisition_function(datasets, models)
        if pending_points is not None and len(pending_points) != 0:
            acq = self._update_penalization(acq, datasets[OBJECTIVE], models[OBJECTIVE], pending_points)

        return acq

    def update_acquisition_function(
        self,
        function: AcquisitionFunction,
        models,
        datasets = None,
        pending_points = None,
        new_optimization_step: bool = True,
    ) -> AcquisitionFunction:
        """
        :param function: The acquisition function to update.
        :param model: The model.
        :param dataset: The data from the observer. Must be populated.
        :param pending_points: Points already chosen to be in the current batch (of shape [M,D]),
            where M is the number of pending points and D is the search space dimension.
        :param new_optimization_step: Indicates whether this call to update_acquisition_function
            is to start of a new optimization step, of to continue collecting batch of points
            for the current step. Defaults to ``True``.
        :return: The updated acquisition function.
        """
        # tf.debugging.Assert(dataset is not None, [])
        # dataset = cast(Dataset, dataset)
        # tf.debugging.assert_positive(len(dataset), message="Dataset must be populated.")
        # tf.debugging.Assert(self._base_acquisition_function is not None, [])

        if new_optimization_step:
            self._update_base_acquisition_function(datasets, models)

        if pending_points is None or len(pending_points) == 0:
            # no penalization required if no pending_points
            return cast(AcquisitionFunction, self._base_acquisition_function)

        return self._update_penalization(function, datasets[OBJECTIVE], models[OBJECTIVE], pending_points)

    def _update_penalization(
        self,
        function,
        dataset: Dataset,
        model,
        pending_points = None,
    ) -> AcquisitionFunction:
        tf.debugging.assert_rank(pending_points, 2)
        if self._penalized_acquisition is not None and isinstance(
            self._penalization, UpdatablePenalizationFunction
        ):
            # if possible, just update the penalization function variables
            self._penalization.update(pending_points, self._lipschitz_constant, self._eta)
            return self._penalized_acquisition
        else:
            # otherwise construct a new penalized acquisition function
            self._penalization = self._lipschitz_penalizer(
                model, pending_points, self._lipschitz_constant, self._eta
            )

            # @tf.function
            def penalized_acquisition(x: TensorType) -> TensorType:
                log_acq = tf.math.log(
                    cast(AcquisitionFunction, self._base_acquisition_function)(x)
                ) + tf.math.log(cast(PenalizationFunction, self._penalization)(x))
                return tf.math.exp(log_acq)
            self._penalized_acquisition = penalized_acquisition
            return penalized_acquisition
    @tf.function(experimental_relax_shapes=True)
    def _get_lipschitz_estimate(
        self, model, sampled_points: TensorType
    ) -> Tuple[TensorType, TensorType]:
        with tf.GradientTape() as g:
            g.watch(sampled_points)
            mean, _ = model.predict(sampled_points)
        grads = g.gradient(mean, sampled_points)
        grads_norm = tf.norm(grads, axis=1)
        max_grads_norm = tf.reduce_max(grads_norm)
        eta = tf.reduce_min(mean, axis=0)
        return max_grads_norm, eta

    def _update_base_acquisition_function(
        self, datasets: Dataset, models
    ) -> AcquisitionFunction:
        samples = self._search_space.sample(num_samples=self._num_samples)
        samples = tf.concat([datasets[OBJECTIVE].query_points, samples], 0)

        lipschitz_constant, eta = self._get_lipschitz_estimate(models[OBJECTIVE], samples)
        if lipschitz_constant < 1e-5:  # threshold to improve numerical stability for 'flat' models
            lipschitz_constant = 10

        self._lipschitz_constant = lipschitz_constant
        self._eta = eta
        if self._base_acquisition_function is not None:
            self._base_acquisition_function = self._base_builder.update_acquisition_function(
                self._base_acquisition_function,
                models,
                datasets=datasets,
            )
        # elif isinstance(self._base_builder, ExpectedImprovement):  # reuse eta estimate
        #     self._base_acquisition_function = cast(
        #         AcquisitionFunction, expected_improvement(models, self._eta)
        #     )
        else:
            self._base_acquisition_function = self._base_builder.prepare_acquisition_function(
                models,
                datasets=datasets,
            )
        return self._base_acquisition_function