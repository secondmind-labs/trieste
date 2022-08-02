import numpy as np
import pymoo
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from cvxopt import matrix
from cvxopt.solvers import qp
import matplotlib.pyplot as plt
from scipy.stats import norm

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

        # Calculate p matrix
        p = self._calc_p(filtered_mean_std)

        # Calculate q matrix
        q = self.calc_q(p)

        # Optimise (y^T)Py - 1 as we do not have risk free assets
        x_star = self._find_x_star(p, q)

        # Select batch to observe
        batch = self._get_batch(x_star, filtered_points, filtered_mean_std, batch_size)

        return batch

    def filter_points(self, nd_points, nd_mean_std):

        probs_of_improvement = self.pi(nd_points)

        above_threshold = probs_of_improvement > self.pi_threshold

        return nd_points[above_threshold.squeeze(),:], nd_mean_std[above_threshold.squeeze(),:]


    def _calc_p(self, a):

        # First determine suitable l and u values for the reference points

        # According to (the text before) Theorem 1 in Guerreiro and Fonesca (2016)
        #  we can just set l to be the min of each dimension
        l = (float(min(a[:, 0])), float(min(a[:, 1])))

        # Get deltas for u
        u_deltas = (float(max(a[:, 0])) - float(min(a[:, 0]))) * 0.2, (float(max(a[:, 1])) - float(min(a[:, 1]))) * 0.2

        # We need u to be higher than mean or std
        u = (float(max(a[:, 0])) + u_deltas[0] , float(max(a[:, 1])) + u_deltas[1] ) # TODO alex: Check this is the correct thing to do

        # Calculate matrix P

        # Out is square matrix determined by number of points in a
        p = np.zeros([a.shape[0], a.shape[0]])

        # Can calculate denominator for each element upfront
        denominator = (u[0] - l[0]) * (u[1] - l[1])

        # Fill in entries of P
        for i in range(p.shape[0]):
            for j in range(p.shape[0]):
                p[i,j] = ((u[0] - max(a[i, 0],a[j, 0])) * (u[1] - max(a[i, 1], a[j, 1])))

        p /= denominator

        return p

    def calc_q(self, p):

        r = np.expand_dims(np.diagonal(p), axis=1)

        q = p - np.dot(r, np.transpose(r))

        return q


    def _find_x_star(self, p, q):

        n_pts = p.shape[0]

        # We need to find r, for the sum to one condition
        r = np.diagonal(p)

        # Use Quadratic Programming to find y*
        # The cvxopt QP solver solves the problem:
        # minimise (1/2)(x^T)Px + (q^T)x
        # subject to Gx <= h (elementwise)
        #        and  Ax = b

        # The minimisation problem we have is:
        # minimise (y^T)Py
        # subject to y >= 0 (elementwise)
        #        and (r^T)y = 1     

        # Therefore we define the inputs as follows
        # P is already correct
        P = matrix(q)

        # We don't have a q term, so just create zeros in the correct shape
        q = matrix(np.zeros([n_pts, 1]))

        # We want to flip the inequality, and just maintain the values of y with G
        # So we set it to -I
        G = matrix(-1*np.eye(n_pts))

        # We want each element to be greater than zero, so we set h to a zero vector
        h = matrix(np.zeros([n_pts, 1]))

        # We want to sum each element of r*y, so we make A a vector of r
        A = matrix(np.expand_dims(r, axis=0))

        # We want the elements to sum to 1, so b is just 1
        b = matrix(np.ones([1,1]))

        # Now we can perform the optimisation
        optim = qp(P=P,q=q,G=G,h=h,A=A,b=b)

        # Extract y* from the optimiser output
        y_star = np.array(optim["x"])

        # Calculate x*
        x_star = y_star / np.sum(y_star)

        return x_star

    def _get_batch(self, x_star, nd_points, nd_mean_std, batch_size):

        
        sorted_array = self._sort_points_by_x_star(x_star, nd_points, nd_mean_std, verbose=True)

        # Plot points selected on pareto front
        plt.scatter(sorted_array[:,3], sorted_array[:,4], c="b")
        plt.scatter(sorted_array[:batch_size,3], sorted_array[:batch_size,4], c="k")
        plt.show()


        
        if self.replication:
            raise NotImplementedError
        else:
            if batch_size <= sorted_array.shape[0]:
                return sorted_array[:batch_size,1:3]
            else:
                raise ValueError("Batch size greater than number of non-dominated points")


    def _sort_points_by_x_star(self, x_star, nd_points, nd_mean_std, verbose=False):

        x_star_and_points = np.concatenate([x_star, nd_points, nd_mean_std], axis=1)
        sorted_array = x_star_and_points[x_star_and_points[:,0].argsort()[::-1]]
        return sorted_array