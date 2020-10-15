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

from ..datasets import Dataset
from ..type import QueryPoints
from ..models import ModelInterface
from ..space import SearchSpace

from scipy.optimize import bisect

AcquisitionFunction = Callable[[QueryPoints], tf.Tensor]
""" Type alias for acquisition functions. """


class AcquisitionFunctionBuilder(ABC):
    """ An :class:`AcquisitionFunctionBuilder` builds an acquisition function. """

    @abstractmethod
    def prepare_acquisition_function(
        self, datasets: Mapping[str, Dataset], models: Mapping[str, ModelInterface]
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
                self, datasets: Mapping[str, Dataset], models: Mapping[str, ModelInterface]
            ) -> AcquisitionFunction:
                return single_builder.prepare_acquisition_function(datasets[tag], models[tag])

            def __repr__(self) -> str:
                return f"{single_builder!r} using tag {tag!r}"

        return _Anon()

    @abstractmethod
    def prepare_acquisition_function(
        self, dataset: Dataset, model: ModelInterface
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
        self, dataset: Dataset, model: ModelInterface
    ) -> AcquisitionFunction:
        """
        :param dataset: Unused.
        :param model: The model over the specified ``dataset``.
        :return: The expected improvement function.
        """
        mean, _ = model.predict(dataset.query_points)
        eta = tf.reduce_min(mean, axis=0)
        return lambda at: self._acquisition_function(model, eta, at)

    @staticmethod
    def _acquisition_function(model: ModelInterface, eta: tf.Tensor, at: QueryPoints) -> tf.Tensor:
        return expected_improvement(model, eta, at)


def expected_improvement(model: ModelInterface, eta: tf.Tensor, at: QueryPoints) -> tf.Tensor:
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


class MaxValueEntropySearch(SingleModelAcquisitionBuilder):
    """
    Builder for the max-value entropy search acquisition function (for function minimisation)
    """

    def __init__(self,search_space: SearchSpace, num_samples: int = 10, grid_size: int = 5000):
        """
        :param search_space: The search space over which the acquisiiton function is defined.
        :param num_samples: Number of sample draws of the minimal value.
        :param grid_size: Size of random grid used to fit the gumbel distribution 
            (recommend scaling with search space dimension).
        """
        self._search_space = search_space

        if num_samples <= 0:
            raise ValueError(
                f"num_samples must be positive, got {num_samples}"
            )
        self._num_samples = num_samples

        if grid_size <= 0:
            raise ValueError(
                f"num_samples must be positive, got {grid_size}"
            )
        self._grid_size = grid_size

    def prepare_acquisition_function(
        self, dataset: Dataset, model: ModelInterface
    ) -> AcquisitionFunction:
        """
        Need to sample possible min-values from our posterior. 
        To do this we implement a Gumbel sampler.
        We approximate Pr(y*^hat<y) by Gumbel(a,b) then sample from Gumbel.

        :param dataset: Unused.
        :param model: The model over the specified ``dataset``.
        :return: The MES function.
        """
        query_points = self._search_space.sample(self._grid_size)
        query_points = tf.concat([dataset.query_points,query_points],0)
        fmean, fvar = model.predict(query_points)
        fsd = tf.math.sqrt(fvar)

        def probf(x: tf.Tensor) -> tf.Tensor :  # Build empirical CDF
            unit_normal = tfp.distributions.Normal(tf.cast(0, tf.float64), tf.cast(1, tf.float64))
            log_cdf = unit_normal.log_cdf(- (x - fmean) / fsd)
            return tf.exp(tf.reduce_sum(log_cdf, axis=0))

        left = tf.reduce_min(fmean - 5 * fsd)
        right = tf.reduce_max(fmean + 5 * fsd)

        def binary_search(val: float) -> float:  # Fit Gumbel quantiles
            return bisect(lambda x: probf(x) - val, left, right, maxiter=10000, xtol=0.00001)
        
        q1, med, q2 = map(binary_search, [0.25, 0.5, 0.75])

        b = (q1 - q2) / (tf.math.log(tf.math.log(4. / 3.)) - tf.math.log(tf.math.log(4.)))
        a = med + b * tf.math.log(tf.math.log(2.))

        uniform_samples = tf.random.uniform([self._num_samples],dtype=tf.dtypes.float64)
        gumbel_samples = -tf.math.log(-tf.math.log(uniform_samples)) * tf.cast(b, tf.float64) + tf.cast(a,tf.float64)

        return lambda at: self._acquisition_function(model, gumbel_samples, at)

    @staticmethod
    def _acquisition_function(model: ModelInterface, samples: tf.Tensor, at: QueryPoints) -> tf.Tensor:
        return max_value_entropy_search(model, samples, at)


def max_value_entropy_search(model: ModelInterface, samples: tf.Tensor, at: QueryPoints) -> tf.Tensor:
    r"""
    Computes the information gain, i.e the change in entropy of p_min if we would evaluate x.

    See the following for details:

    ::

        @article{wang2017max,
          title={Max-value entropy search for efficient Bayesian optimization},
          author={Wang, Zi and Jegelka, Stefanie},
          journal={arXiv preprint arXiv:1703.01968},
          year={2017}
        }

    :param model: The model of the objective function.
    :param samples: Samples from p_min
    :param at: The points for which to calculate the expected improvement.
    :return: The expected improvement at ``at``.
    """
    fmean, fvar = model.predict(at)
    fsd = tf.math.sqrt(fvar)
    fsd = tf.clip_by_value(fsd, 1.0e-8, tf.float64.max) # clip below to improve numerical stability
    
    normal = tfp.distributions.Normal(tf.cast(0,tf.float64), tf.cast(1,tf.float64))
    gamma = (samples - fmean) / fsd

    minus_cdf = 1 - normal.cdf(gamma)
    minus_cdf = tf.clip_by_value(minus_cdf, 1.0e-8, 1) # clip below to improve numerical stability
    f_acqu_x = -gamma * normal.prob(gamma) / (2 * minus_cdf) - tf.math.log(minus_cdf)
    f_acqu_x = tf.math.reduce_mean(f_acqu_x, axis=1)
        
    return tf.reshape(f_acqu_x,[-1, 1])


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
        self, dataset: Dataset, model: ModelInterface
    ) -> AcquisitionFunction:
        """
        :param dataset: Unused.
        :param model: The model over the specified ``dataset``.
        :return: The negative of the lower confidence bound function. This function will raise
            `ValueError` if ``beta`` is negative.
        """
        return lambda at: self._acquisition_function(model, self._beta, at)

    @staticmethod
    def _acquisition_function(model: ModelInterface, beta: float, at: QueryPoints) -> tf.Tensor:
        return -lower_confidence_bound(model, beta, at)


class NegativePredictiveMean(NegativeLowerConfidenceBound):
    """
    Builder for the negative of the predictive mean. The predictive mean is minimised on minimising
    the objective function. The negative predictive mean is therefore maximised.
    """

    def __init__(self):
        super().__init__(beta=0.0)


def lower_confidence_bound(model: ModelInterface, beta: float, at: QueryPoints) -> tf.Tensor:
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
        self, dataset: Dataset, model: ModelInterface
    ) -> AcquisitionFunction:
        """
        :param dataset: Unused.
        :param model: The model over the specified ``dataset``.
        :return: The probability of feasibility acquisition function.
        """
        return lambda at: self._acquisition_function(model, self.threshold, at)

    @staticmethod
    def _acquisition_function(
        model: ModelInterface, threshold: Union[float, tf.Tensor], at: QueryPoints
    ) -> tf.Tensor:
        return probability_of_feasibility(model, threshold, at)


def probability_of_feasibility(
    model: ModelInterface, threshold: Union[float, tf.Tensor], at: QueryPoints
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
