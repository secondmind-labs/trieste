# Copyright 2020 The Trieste Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from abc import ABC, abstractmethod
from typing import Callable, Mapping, Union

import tensorflow as tf
import tensorflow_probability as tfp
from typing_extensions import final

from ..data import Dataset
from ..type import QueryPoints
from ..models import ProbabilisticModel


AcquisitionFunction = Callable[[QueryPoints], tf.Tensor]
"""
Type alias for acquisition functions.


An `AcquisitionFunction` maps a single query point (of dimension `D`) to a single value that
describes how useful it would be evaluate that point (to our goal of optimizing the objective
function). Thus, with leading dimensions, an `AcquisitionFunction` takes input shape
`[..., D]` and returns shape `[..., 1]`.

**Note:** Type checkers will not be able to distinguish an `AcquisitionFunction` from a
`BatchAcquisitionFunction`.
"""

BatchAcquisitionFunction = Callable[[QueryPoints], tf.Tensor]
""" 
Type alias for batch acquisition functions. 

A `BatchAcquisitionFunction` maps a set of `B` query points (each of dimension `D`) to a single
value that describes how useful it would be evaluate all these points together (to our goal of
optimizing the objective function). Thus, with leading dimensions, a `BatchAcquisitionFunction`
takes input shape `[..., B, D]` and returns shape `[..., 1]`.

**Note:** Type checkers will not be able to distinguish an `AcquisitionFunction` from a
`BatchAcquisitionFunction`.
"""


class AcquisitionFunctionBuilder(ABC):
    """ An :class:`AcquisitionFunctionBuilder` builds an acquisition function. """

    @abstractmethod
    def prepare_acquisition_function(
        self, datasets: Mapping[str, Dataset], models: Mapping[str, ProbabilisticModel]
    ) -> AcquisitionFunction:
        """
        :param datasets: The data from the observer.
        :param models: The models over each dataset in ``datasets``.
        :return: An acquisition function.
        """


class SingleModelAcquisitionBuilder(ABC):
    """
    Convenience acquisition function builder for an acquisition function (or component of a
    composite acquisition function) that requires only one model, dataset pair.
    """

    def using(self, tag: str) -> AcquisitionFunctionBuilder:
        """
        :param tag: The tag for the model, dataset pair to use to build this acquisition function.
        :return: An acquisition function builder that selects the model and dataset specified by
            ``tag``, as defined in :meth:`prepare_acquisition_function`.
        """
        single_builder = self

        class _Anon(AcquisitionFunctionBuilder):
            def prepare_acquisition_function(
                self, datasets: Mapping[str, Dataset], models: Mapping[str, ProbabilisticModel]
            ) -> AcquisitionFunction:
                return single_builder.prepare_acquisition_function(datasets[tag], models[tag])

            def __repr__(self) -> str:
                return f"{single_builder!r} using tag {tag!r}"

        return _Anon()

    @abstractmethod
    def prepare_acquisition_function(
        self, dataset: Dataset, model: ProbabilisticModel
    ) -> AcquisitionFunction:
        """
        :param dataset: The data to use to build the acquisition function.
        :param model: The model over the specified ``dataset``.
        :return: An acquisition function.
        """


class ExpectedImprovement(SingleModelAcquisitionBuilder):
    """
    Builder for the expected improvement function where the "best" value is taken to be the minimum
    of the posterior mean at observed points.
    """

    def prepare_acquisition_function(
        self, dataset: Dataset, model: ProbabilisticModel
    ) -> AcquisitionFunction:
        """
        :param dataset: The data from the observer. Must be populated.
        :param model: The model over the specified ``dataset``.
        :return: The expected improvement function.
        :raise ValueError: If ``dataset`` is empty.
        """
        if len(dataset.query_points) == 0:
            raise ValueError("Dataset must be populated.")

        mean, _ = model.predict(dataset.query_points)
        eta = tf.reduce_min(mean, axis=0)
        return lambda at: self._acquisition_function(model, eta, at)

    @staticmethod
    def _acquisition_function(
        model: ProbabilisticModel, eta: tf.Tensor, at: QueryPoints
    ) -> tf.Tensor:
        return expected_improvement(model, eta, at)


def expected_improvement(model: ProbabilisticModel, eta: tf.Tensor, at: QueryPoints) -> tf.Tensor:
    r"""
    The Expected Improvement (EI) acquisition function for single-objective global optimization.
    Return the expectation of the improvement at ``at`` over the current "best" observation ``eta``,
    where an improvement moves towards the objective function's minimum, and the expectation is
    calculated with respect to the ``model`` posterior. For model posterior :math:`f`, this is

    .. math:: x \mapsto \mathbb E \left[ \max (\eta - f(x), 0) \right]

    This function was introduced by Mockus et al, 1975. See the following for details:

    ::

       @article{Jones:1998,
            title={Efficient global optimization of expensive black-box functions},
            author={Jones, Donald R and Schonlau, Matthias and Welch, William J},
            journal={Journal of Global optimization},
            volume={13},
            number={4},
            pages={455--492},
            year={1998},
            publisher={Springer}
       }

    :param model: The model of the objective function.
    :param eta: The "best" observation.
    :param at: The points for which to calculate the expected improvement.
    :return: The expected improvement at ``at``.
    """
    mean, variance = model.predict(at)
    normal = tfp.distributions.Normal(mean, tf.sqrt(variance))
    return (eta - mean) * normal.cdf(eta) + variance * normal.prob(eta)


class NegativeLowerConfidenceBound(SingleModelAcquisitionBuilder):
    """
    Builder for the negative of the lower confidence bound. The lower confidence bound is typically
    minimised, so the negative is suitable for maximisation.
    """

    def __init__(self, beta: float = 1.96):
        """
        :param beta: Weighting given to the variance contribution to the lower confidence bound.
            Must not be negative.
        """
        self._beta = beta

    def prepare_acquisition_function(
        self, dataset: Dataset, model: ProbabilisticModel
    ) -> AcquisitionFunction:
        """
        :param dataset: Unused.
        :param model: The model over the specified ``dataset``.
        :return: The negative of the lower confidence bound function. This function will raise
            `ValueError` if ``beta`` is negative.
        """
        return lambda at: self._acquisition_function(model, self._beta, at)

    @staticmethod
    def _acquisition_function(model: ProbabilisticModel, beta: float, at: QueryPoints) -> tf.Tensor:
        return -lower_confidence_bound(model, beta, at)


class NegativePredictiveMean(NegativeLowerConfidenceBound):
    """
    Builder for the negative of the predictive mean. The predictive mean is minimised on minimising
    the objective function. The negative predictive mean is therefore maximised.
    """

    def __init__(self):
        super().__init__(beta=0.0)


def lower_confidence_bound(model: ProbabilisticModel, beta: float, at: QueryPoints) -> tf.Tensor:
    r"""
    The lower confidence bound (LCB) acquisition function for single-objective global optimization.

    .. math:: x^* \mapsto \mathbb{E} [f(x^*)|x, y] - \beta \sqrt{ \mathrm{Var}[f(x^*)|x, y] }

    See the following for details:

    ::

        @inproceedings{Srinivas:2010,
            author = "Srinivas, Niranjan and Krause, Andreas and Seeger, Matthias and Kakade, Sham M.",
            booktitle = "{Proceedings of the 27th International Conference on Machine Learning (ICML-10)}",
            editor = "F{\"u}rnkranz, Johannes and Joachims, Thorsten",
            pages = "1015--1022",
            publisher = "Omnipress",
            title = "{Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design}",
            year = "2010"
        }

    :param model: The model of the objective function.
    :param beta: The weight to give to the standard deviation contribution of the LCB. Must not be
        negative.
    :param at: The points at which to evaluate the LCB.
    :return: The lower confidence bound at ``at``.
    :raise ValueError: If ``beta`` is negative.
    """
    if beta < 0:
        raise ValueError(
            f"Standard deviation scaling parameter beta must not be negative, got {beta}"
        )

    mean, variance = model.predict(at)
    return mean - beta * tf.sqrt(variance)


class ProbabilityOfFeasibility(SingleModelAcquisitionBuilder):
    r"""
    Builder for the :func:`probability_of_feasibility` acquisition function, defined in
    Garner, 2014 as

    .. math:: \int_{-\infty}^{\tau} p(c(\mathbf{x}) | \mathbf{x}, \mathcal{D}) \mathrm{d} c(\mathbf{x}) \qquad ,

    where :math:`\tau` is a threshold. Values below the threshold are considered feasible by the
    constraint function. See the following for details:

    ::

        @inproceedings{gardner14,
            title={Bayesian Optimization with Inequality Constraints},
            author={Jacob Gardner and Matt Kusner and Zhixiang and Kilian Weinberger and John Cunningham},
            booktitle={Proceedings of the 31st International Conference on Machine Learning},
            year={2014},
            volume={32},
            number={2},
            series={Proceedings of Machine Learning Research},
            month={22--24 Jun},
            publisher={PMLR},
            url={http://proceedings.mlr.press/v32/gardner14.html},
        }

        @article{schonlau1998global,
            title={Global versus local search in constrained optimization of computer models},
            author={Schonlau, Matthias and Welch, William J and Jones, Donald R},
            journal={Lecture Notes-Monograph Series},
            pages={11--25},
            year={1998},
            publisher={JSTOR}
        }

    """

    def __init__(self, threshold: Union[float, tf.Tensor]):
        """
        :param threshold: The (scalar) probability of feasibility threshold.
        :raise ValueError (or InvalidArgumentError): If ``threshold`` is not a scalar.
        """
        tf.debugging.assert_scalar(threshold)

        super().__init__()

        self._threshold = threshold

    @property
    def threshold(self) -> Union[float, tf.Tensor]:
        """ The probability of feasibility threshold. """
        return self._threshold

    def prepare_acquisition_function(
        self, dataset: Dataset, model: ProbabilisticModel
    ) -> AcquisitionFunction:
        """
        :param dataset: Unused.
        :param model: The model over the specified ``dataset``.
        :return: The probability of feasibility acquisition function.
        """
        return lambda at: self._acquisition_function(model, self.threshold, at)

    @staticmethod
    def _acquisition_function(
        model: ProbabilisticModel, threshold: Union[float, tf.Tensor], at: QueryPoints
    ) -> tf.Tensor:
        return probability_of_feasibility(model, threshold, at)


def probability_of_feasibility(
    model: ProbabilisticModel, threshold: Union[float, tf.Tensor], at: QueryPoints
) -> tf.Tensor:
    r"""
    The probability of feasibility acquisition function defined in Garner, 2014 as

    .. math:: \int_{-\infty}^{\tau} p(c(\mathbf{x}) | \mathbf{x}, \mathcal{D}) \mathrm{d}c(\mathbf{x}) \qquad ,

    where :math:`\tau` is a threshold. Values below the threshold are considered feasible by the
    constraint function.

    :param model: The model of the objective function.
    :param threshold: The (scalar) probability of feasibility threshold.
    :param at: The points at which to evaluate the probability of feasibility. Must have rank at
        least two
    :return: The probability of feasibility at ``at``.
    :raise ValueError (or InvalidArgumentError): If arguments have the incorrect shape.
    """
    tf.debugging.assert_scalar(threshold)
    tf.debugging.assert_rank_at_least(at, 2)

    mean, var = model.predict(at)
    distr = tfp.distributions.Normal(mean, tf.sqrt(var))
    return distr.cdf(tf.cast(threshold, at.dtype))


class ExpectedConstrainedImprovement(AcquisitionFunctionBuilder):
    """
    Builder for the _expected constrained improvement_ acquisition function defined in
    Gardner, 2014. The acquisition function computes the expected improvement from the best
    feasible point, where feasible points are those that (probably) satisfy some constraint. Where
    there are no feasible points, this builder simply builds the constraint function.

    See the following for details:

    ::

        @inproceedings{gardner14,
            title={Bayesian Optimization with Inequality Constraints},
            author={Jacob Gardner and Matt Kusner and Zhixiang and Kilian Weinberger and John Cunningham},
            booktitle={Proceedings of the 31st International Conference on Machine Learning},
            year={2014},
            volume={32},
            number={2},
            series={Proceedings of Machine Learning Research},
            month={22--24 Jun},
            publisher={PMLR},
            url={http://proceedings.mlr.press/v32/gardner14.html},
        }

    """

    def __init__(
        self,
        objective_tag: str,
        constraint_builder: AcquisitionFunctionBuilder,
        min_feasibility_probability: Union[float, tf.Tensor] = 0.5,
    ):
        """
        :param objective_tag: The tag for the objective data and model.
        :param constraint_builder: The builder for the constraint function.
        :param min_feasibility_probability: The minimum probability of feasibility for a
            "best point" to be considered feasible.
        :raise ValueError (or InvalidArgumentError): If ``min_feasibility_probability`` is not a
            scalar in the unit interval :math:`[0, 1]`.
        """
        tf.debugging.assert_scalar(min_feasibility_probability)

        if not 0 <= min_feasibility_probability <= 1:
            raise ValueError(
                f"Minimum feasibility probability must be between 0 and 1 inclusive,"
                f" got {min_feasibility_probability}"
            )

        self._objective_tag = objective_tag
        self._constraint_builder = constraint_builder
        self._min_feasibility_probability = min_feasibility_probability

    def prepare_acquisition_function(
        self, datasets: Mapping[str, Dataset], models: Mapping[str, ProbabilisticModel]
    ) -> AcquisitionFunction:
        """
        :param datasets: The data from the observer.
        :param models: The models over each dataset in ``datasets``.
        :return: The expected constrained improvement acquisition function.
        :raise KeyError: If `objective_tag` is not found in ``datasets`` and ``models``.
        :raise ValueError: If the objective data is empty.
        """
        objective_model = models[self._objective_tag]
        objective_dataset = datasets[self._objective_tag]

        if len(objective_dataset) == 0:
            raise ValueError(
                "Expected improvement is defined with respect to existing points in the objective"
                " data, but the objective data is empty."
            )

        constraint_fn = self._constraint_builder.prepare_acquisition_function(datasets, models)
        pof = constraint_fn(objective_dataset.query_points)
        is_feasible = pof >= self._min_feasibility_probability

        if not tf.reduce_any(is_feasible):
            return constraint_fn

        mean, _ = objective_model.predict(objective_dataset.query_points)
        eta = tf.reduce_min(tf.boolean_mask(mean, is_feasible), axis=0)

        return lambda at: expected_improvement(objective_model, eta, at) * constraint_fn(at)


class IndependentReparametrizationSampler:
    r"""
    This sampler employs the *reparameterization trick* to approximate samples from a
    :class:`ProbabilisticModel`\ 's predictive distribution as

    .. math:: x \mapsto \mu(x) + \epsilon \sigma(x)

    where :math:`\epsilon \sim \mathcal N (0, 1)` is constant for a given sampler, thus ensuring
    samples form a continuous curve.
    """

    def __init__(self, sample_size: int, model: ProbabilisticModel):
        """
        :param sample_size: The number of samples to take at each point. Must be positive.
        :param model: The model to sample from.
        :raise ValueError (or InvalidArgumentError): If ``sample_size`` is not positive.
        """
        tf.debugging.assert_positive(sample_size)

        self._sample_size = sample_size
        self._eps = tf.Variable(
            tf.ones([sample_size, 0], dtype=tf.float64), shape=[sample_size, None]
        )  # [S, 0]
        self._model = model

    def sample(self, at: QueryPoints) -> tf.Tensor:
        """
        Return approximate samples from the `model` specified at :meth:`__init__`. Multiple calls to
        :meth:`sample`, for any given :class:`IndependentReparametrizationSampler` and ``at``, will
        produce the exact same samples. Calls to :meth:`sample` on *different*
        :class:`IndependentReparametrizationSampler` instances will produce different samples.

        :param at: Where to sample the predictive distribution, with shape `[..., D]`, for points
            of dimension `D`.
        :return: The samples, of shape `[..., S, L]`, where `S` is the `sample_size` and `L` is the
            number of latent model dimensions.
        :raise ValueError (or InvalidArgumentError): If ``at`` is a scalar.
        """
        tf.debugging.assert_rank_at_least(at, 1)
        mean, cov = self._model.predict(at[..., None, :])  # [..., 1, L], [..., 1, L]

        if tf.size(self._eps) == 0:
            self._eps.assign(
                tf.random.normal([self._sample_size, mean.shape[-1]], dtype=tf.float64)
            )  # [S, L]

        return mean + tf.sqrt(cov) * tf.cast(self._eps, cov.dtype)  # [..., S, L]


class MCIndAcquisitionFunctionBuilder(AcquisitionFunctionBuilder):
    """
    A :class:`MCIndAcquisitionFunctionBuilder` builds an acquisition function that
    estimates the value of evaluating the observer at a given point, and does this using Monte-Carlo
    estimation via the reparameterization trick. This class is essentially a convenience
    :class:`AcquisitionFunctionBuilder` using a :class:`IndependentReparametrizationSampler`.
    """

    def __init__(self, sample_size: int):
        """
        :param sample_size: The number of samples to take at each point. Must be positive.
        :raise ValueError (or InvalidArgumentError): If ``sample_size`` is not positive.
        """
        tf.debugging.assert_positive(sample_size)
        self._sample_size = sample_size

    @final
    def prepare_acquisition_function(
        self, datasets: Mapping[str, Dataset], models: Mapping[str, ProbabilisticModel]
    ) -> AcquisitionFunction:
        samplers = {
            key: IndependentReparametrizationSampler(self._sample_size, model)
            for key, model in models.items()
        }
        return self._build_with_sampler(datasets, models, samplers)

    @abstractmethod
    def _build_with_sampler(
        self,
        datasets: Mapping[str, Dataset],
        models: Mapping[str, ProbabilisticModel],
        samplers: Mapping[str, IndependentReparametrizationSampler],
    ) -> AcquisitionFunction:
        """
        :param datasets: The data from the observer.
        :param models: The models over each dataset in ``datasets``.
        :param samplers: A sampler for each model in ``models``.
        :return: An acquisition function.
        """


class SingleModelMCIndAcquisitionFunctionBuilder(SingleModelAcquisitionBuilder):
    """
    A :class:`SingleModelMCIndAcquisitionFunctionBuilder` builds an acquisition function that
    estimates the value of evaluating the observer at a given point, and does this using Monte-Carlo
    estimation via the reparameterization trick. This class is essentially a convenience
    :class:`AcquisitionFunctionBuilder` using a :class:`IndependentReparametrizationSampler`.

    Subclasses implement :meth:`_build_with_samples` which, in addition to the arguments `dataset`
    and `model`, provides a :class:`IndependentReparametrizationSampler` which can be used to
    approximate continuous samples from the model.
    """

    def __init__(self, sample_size: int):
        """
        :param sample_size: The number of samples to take at each point. Must be positive.
        :raise ValueError (or InvalidArgumentError): If ``sample_size`` is not positive.
        """
        tf.debugging.assert_positive(sample_size)
        self._sample_size = sample_size

    @final
    def prepare_acquisition_function(
        self, dataset: Dataset, model: ProbabilisticModel
    ) -> AcquisitionFunction:
        sampler = IndependentReparametrizationSampler(self._sample_size, model)
        return self._build_with_sampler(dataset, model, sampler)

    @abstractmethod
    def _build_with_sampler(
        self,
        dataset: Dataset,
        model: ProbabilisticModel,
        sampler: IndependentReparametrizationSampler,
    ) -> AcquisitionFunction:
        """
        :param dataset: The data to use to build the acquisition function.
        :param model: The model over the specified ``dataset``.
        :param sampler: A sampler for ``model``.
        :return: An acquisition function.
        """


class BatchAcquisitionFunctionBuilder(ABC):
    """ A :class:`BatchAcquisitionFunctionBuilder` builds a batch acquisition function. """

    @abstractmethod
    def prepare_acquisition_function(
        self, datasets: Mapping[str, Dataset], models: Mapping[str, ProbabilisticModel]
    ) -> BatchAcquisitionFunction:
        """
        :param datasets: The data from the observer.
        :param models: The models over each dataset in ``datasets``.
        :return: A batch acquisition function.
        """
