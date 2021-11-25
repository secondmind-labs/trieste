import tensorflow as tf
import trieste

class SpecialSampler(trieste.acquisition.sampler.BatchReparametrizationSampler):
    def sample(self, at, jitter=trieste.utils.DEFAULTS.JITTER):
        sample = super().sample(at, jitter=jitter)

        gpr_part = sample[:, :, :, 0]
        vgp_part = sample[:, :, :, 1]

        vgp_part = self._model._models[1]._model.likelihood.invlink(vgp_part)

        sample = tf.stack([gpr_part, vgp_part], axis=-1)

        return sample


class SpecialBatchMonteCarloExpectedHypervolumeImprovement(trieste.acquisition.function.AcquisitionFunctionBuilder):
    """ The one in trieste is single model, and we need to pass two models and two datasets
    """

    def __init__(self, sample_size: int,
                 gpr_tag, vgp_tag,
                 jitter: float = trieste.utils.misc.DEFAULTS.JITTER):
        """
        :param sample_size: The number of samples from model predicted distribution for
            each batch of points.
        :param jitter: The size of the jitter to use when stabilising the Cholesky decomposition of
            the covariance matrix.
        :raise ValueError (or InvalidArgumentError): If ``sample_size`` is not positive, or
            ``jitter`` is negative.
        """
        tf.debugging.assert_positive(sample_size)
        tf.debugging.assert_greater_equal(jitter, 0.0)

        super().__init__()

        self._sample_size = sample_size
        self._jitter = jitter

        self._gpr_tag = gpr_tag
        self._vgp_tag = vgp_tag

    def prepare_acquisition_function(
        self,
        models,
        datasets,
    ):
        # VGP dataset will have all points
        # while GPR only successful ones
        query_points = datasets[self._vgp_tag].query_points
        
        # [0] is because we only need mean and not variance
        means = tf.concat([models[self._gpr_tag].predict(query_points)[0], models[self._vgp_tag].predict_y(query_points)[0]], axis=-1)
        _pf = trieste.acquisition.multi_objective.pareto.Pareto(means)
        _reference_pt = trieste.acquisition.multi_objective.pareto.get_reference_point(_pf.front)
        # prepare the partitioned bounds of non-dominated region for calculating of the
        # hypervolume improvement in this area
        _partition_bounds = trieste.acquisition.multi_objective.partition.prepare_default_non_dominated_partition_bounds(_reference_pt, _pf.front)

        models = [(models[self._gpr_tag], 1), (models[self._vgp_tag], 1)]
        sampler = SpecialSampler(self._sample_size, trieste.models.interfaces.ModelStack(*models))

        return trieste.acquisition.function.batch_ehvi(sampler, self._jitter, _partition_bounds)


def create_empty_dataset(search_space):
    return trieste.data.Dataset(
                tf.zeros((0, search_space.dimension), tf.float64),
                tf.zeros((0, 1), tf.float64)
            )


def collect_initial_points(search_space, observer, min_points=1, print_size=True):
    num_initial_points = min_points
    initial_query_points = search_space.sample(num_initial_points)
    initial_data = observer(initial_query_points)

    # collect points until we have at least `min_points` in each dataset
    while any(len(initial_data[tag]) < min_points for tag in initial_data):
        initial_query_points = search_space.sample(1)
        new_initial_data = observer(initial_query_points)
        for tag in initial_data:
            initial_data[tag] = initial_data[tag] + new_initial_data[tag]
        num_initial_points += 1

    if print_size:
        for tag in initial_data:
            print(f"Points in dataset {tag}: {len(initial_data[tag])}")

    return num_initial_points, initial_data


from trieste.acquisition.multi_objective.dominance import non_dominated

def find_pf_points(all_query_points, objective_model_values, failure_model_values):
    pf_points, _ = non_dominated(tf.concat([objective_model_values, failure_model_values], axis=1))
    pf_input_points = []
    for pf_point in pf_points:
        pf_input_point = tf.boolean_mask(all_query_points, tf.equal(objective_model_values, pf_point[0])[:,0])
        if len(pf_input_point) > 1:
            pf_input_point = pf_input_point[0:1, :]
        pf_input_points.append(pf_input_point)
    pf_input_points = tf.concat(pf_input_points, axis=0)

    return pf_input_points, pf_points