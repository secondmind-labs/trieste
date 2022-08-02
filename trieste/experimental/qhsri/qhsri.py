import numpy as np
import pymoo
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from cvxopt import matrix
from cvxopt.solvers import qp
import matplotlib.pyplot as plt
from scipy.stats import norm
from trieste.acquisition.multi_objective import Pareto

class ProbabilityOfImprovement:

    def __init__(self, model, current_best):

        self.current_best = current_best
        self.model = model

    def __call__(self, x):
        
        mean, var = self.model.predict_y(x)
        mean, std = np.array(mean), np.sqrt(np.array(var))
        pi = norm.cdf((self.current_best - mean)/std)

        return pi

    def update(self, new_model, new_best):

        self.model = new_model
        self.current_best = new_best


class MeanStdTradeoff(pymoo.core.problem.Problem):
    def __init__(self, probabilistic_model):
        super().__init__(n_var=2, n_obj=2, n_constr=0, xl=np.array([0,0]), xu=np.array([1,1]))
        self.probabilistic_model = probabilistic_model

    def _evaluate(self, x, out, *args, **kwargs):
        mean,var = self.probabilistic_model.predict_y(x)
        # Flip sign on std so that minimising is increasing std
        std = -1*np.sqrt(np.array(var))
        out["F"] = np.concatenate([np.array(mean), std], axis=1)

class BatchHypervolumeSharpeRatioIndicator:

    def __init__(self, model, current_best, population_size:int =500, replication=False, pi_threshold = 0.3):

        self.population_size = population_size
        self.replication = replication
        self.model = model
        self.current_best = current_best
        self.pi_threshold = pi_threshold

        self.pi = ProbabilityOfImprovement(self.model, self.current_best)

    def _find_non_dominated_points(self, model, plot=False):
        """Uses NSGA-II to find high-quality non-dominated points
        """

        problem = MeanStdTradeoff(model)
        algorithm = NSGA2(pop_size=self.population_size)
        res = minimize(problem, algorithm, ('n_gen', 200), seed=1, verbose=False)
        if plot:
            plt.scatter(res.X[:,0], res.X[:,1])
            plt.show()
            plt.scatter(res.F[:,0], res.F[:,1], c="r")

        return res.X, res.F

    
    def __call__(self, model, batch_size):

        # Find non-dominated points in the mean/std space
        nd_points, nd_mean_std = self._find_non_dominated_points(model, True)

        # Filter based on probability of improvement
        print(f"Points pre-filtering = {len(nd_points)}")
        filtered_points, filtered_mean_std = self.filter_points(nd_points, nd_mean_std)
        print(f"Points post-filtering = {len(filtered_points)}")

        # set up Pareto set
        pareto_set = Pareto(filtered_mean_std, already_non_dominated=True)

        _, batch_ids = pareto_set.sample(batch_size)

        batch = filtered_points[batch_ids]

        return batch

    def filter_points(self, nd_points, nd_mean_std):

        probs_of_improvement = self.pi(nd_points)

        above_threshold = probs_of_improvement > self.pi_threshold

        return nd_points[above_threshold.squeeze(),:], nd_mean_std[above_threshold.squeeze(),:]