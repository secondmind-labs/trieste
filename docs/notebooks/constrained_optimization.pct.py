# %% [markdown]
# # Constrained Acquisition Function Optimization with Expected Improvement

# %% [markdown]
# Sometimes it is practically convenient to query several points at a time. This notebook demonstrates four ways to perfom batch Bayesian optimization with Trieste.

# %%
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from trieste.experimental.plotting import plot_acq_function_2d
import matplotlib.pyplot as plt
import trieste
from trieste.objectives.utils import mk_observer
from trieste.observer import OBJECTIVE
from trieste.space import Box
from trieste.models.gpflow import GaussianProcessRegression, build_gpr
from trieste.acquisition.function import ExpectedImprovement, ExpectedConstrainedImprovement
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.acquisition.optimizer import generate_continuous_optimizer, NUM_SAMPLES_MIN, NUM_SAMPLES_DIM, NUM_RUNS_DIM
from trieste.experimental.plotting import plot_function_plotly, plot_function_2d
from trieste.utils import Timer
from trieste.acquisition.interface import AcquisitionFunction, SingleModelAcquisitionBuilder, AcquisitionFunctionBuilder
from trieste.models import ProbabilisticModel
from trieste.models.interfaces import ProbabilisticModelType
from trieste.data import Dataset
from trieste.types import TensorType
from scipy.optimize import LinearConstraint
from typing import Optional, Mapping, cast
import gpflow
import textwrap
import ipywidgets
from IPython. display import clear_output
import imageio

np.random.seed(5678)
tf.random.set_seed(5678)

# %% [markdown]
# ## Constraints functions and classes

# %%
ctol = 1e-7

#def constraints_residual(constraints, x):
#    #print('x:', x.shape)
#    return np.concatenate(np.array([[constraint.A@x.T - constraint.lb[..., np.newaxis], constraint.ub[..., np.newaxis] - constraint.A@x.T]
#                                    for constraint in constraints]))
#
def constraints_residual(constraints, x):
    return tf.concat([[tf.linalg.matmul(constraint.A, x, transpose_b=True) - constraint.lb[..., tf.newaxis],
                                constraint.ub[..., tf.newaxis] - tf.linalg.matmul(constraint.A, x, transpose_b=True)]
                      for constraint in constraints], axis=0)

def constraints_satisfied(constraints, x):
    #res_lo, res_up = constraints_tr.residual(x.T)
    #return np.logical_and(np.all(res_lo >= 0., aixs=0), np.all(res_up >= 0., aixs=0))
    #rl, ru = constraints_residual(x)
    #return np.logical_and(np.all(rl >= 0., axis=0), np.all(ru >= 0., axis=0))
    return np.all(np.all(constraints_residual(constraints, x) >= -ctol, axis=0), axis=0)

def constraints_fn(constraints, x):
    #print('x:', x.shape)
    if len(x.shape) == 1:
        x = np.expand_dims(x, axis=0)
    return np.reshape(constraints_residual(constraints, x), (-1, x.shape[0]))

def constraints_jac(constraints, x):
    if len(x.shape) == 1:
        x = np.expand_dims(x, axis=0)
    jac = np.concatenate(np.array([[constraint.A, -constraint.A] for constraint in constraints]))
    return np.reshape(jac, (-1, x.shape[-1], 1))

def constraints_to_dict(constraints, search_space):
    return [{'type': 'ineq',
             'fun': lambda x, i=i, j=j: constraints_fn([constraints[i]], x)[j].squeeze(),
             'jac': lambda x, i=i, j=j: constraints_jac([constraints[i]], x)[j].squeeze(),
            }
            for i in range(len(constraints)) for j in range(len(search_space.lower) * 2)]


# %%
class SimpleConstraintsFeasibility(SingleModelAcquisitionBuilder[ProbabilisticModel]):
    def __init__(self, constraints):
        self._constraints = constraints

    def __repr__(self) -> str:
        """"""
        return "SimpleConstraintsFeasibility()"

    def prepare_acquisition_function(
        self,
        model: ProbabilisticModel,
        dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        """
        :param model: Unused.
        :param dataset: Unused.
        :return: The probability of feasibility function. This function will raise
            :exc:`ValueError` or :exc:`~tf.errors.InvalidArgumentError` if used with a batch size
            greater than one.
        """
        return simple_constraints_feasibility(self._constraints)

    def update_acquisition_function(
        self,
        function: AcquisitionFunction,
        model: ProbabilisticModel,
        dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        """
        :param function: The acquisition function to update.
        :param model: Unused.
        :param dataset: Unused.
        """
        return function  # no need to update anything

def simple_constraints_feasibility(
    constraints
) -> AcquisitionFunction:
    r"""
    :return: The probability of feasibility function. This function will raise
        :exc:`ValueError` or :exc:`~tf.errors.InvalidArgumentError` if used with a batch size
        greater than one.
    :raise ValueError or tf.errors.InvalidArgumentError: If ``threshold`` is not a scalar.
    """

    @tf.function
    def acquisition(x: TensorType) -> TensorType:
        tf.debugging.assert_shapes(
            [(x, [..., 1, None])],
            message="This acquisition function only supports batch sizes of one.",
        )

        c = constraints_residual(constraints, tf.squeeze(x, -2))
        distr = tfp.distributions.Normal(tf.cast(0, x.dtype), tf.cast(1e-3, x.dtype))
        return tf.expand_dims(tf.math.reduce_prod(distr.cdf(c), axis=[0, 1]), axis=-1)

    return acquisition


class ExpectedSimpleConstrainedImprovement(ExpectedConstrainedImprovement[ProbabilisticModel]):
    def __init__(
        self,
        objective_tag: str,
        constraint_builder: AcquisitionFunctionBuilder[ProbabilisticModelType],
        min_feasibility_probability: float | TensorType = 0.5,
    ):
        """
        :param objective_tag: The tag for the objective data and model.
        :param constraint_builder: The builder for the constraint function.
        :param min_feasibility_probability: The minimum probability of feasibility for a
            "best point" to be considered feasible.
        :raise ValueError (or tf.errors.InvalidArgumentError): If ``min_feasibility_probability``
            is not a scalar in the unit interval :math:`[0, 1]`.
        """
        super().__init__(objective_tag, constraint_builder, min_feasibility_probability)

    def __repr__(self) -> str:
        """"""
        return (
            f"ExpectedSimpleConstrainedImprovement({self._objective_tag!r}, {self._constraint_builder!r},"
            f" {self._min_feasibility_probability!r})"
        )

    def prepare_acquisition_function(
        self,
        models: Mapping[str, ProbabilisticModelType],
        datasets: Optional[Mapping[str, Dataset]] = None,
    ) -> AcquisitionFunction:
        """
        :param models: The models over each tag.
        :param datasets: The data from the observer.
        :return: The expected constrained improvement acquisition function. This function will raise
            :exc:`ValueError` or :exc:`~tf.errors.InvalidArgumentError` if used with a batch size
            greater than one.
        :raise KeyError: If `objective_tag` is not found in ``datasets`` and ``models``.
        :raise tf.errors.InvalidArgumentError: If the objective data is empty.
        """
        acquistion_function = super().prepare_acquisition_function(models, datasets)
        #return self._expected_improvement_fn if acquistion_function == self._constrained_improvement_fn else acquistion_function
        tf.debugging.Assert(self._expected_improvement_fn is not None, [tf.constant([])])
        return self._expected_improvement_fn
        
    def update_acquisition_function(
        self,
        function: AcquisitionFunction,
        models: Mapping[str, ProbabilisticModelType],
        datasets: Optional[Mapping[str, Dataset]] = None,
    ) -> AcquisitionFunction:
        """
        :param function: The acquisition function to update.
        :param models: The models for each tag.
        :param datasets: The data from the observer.
        """
        acquistion_function = super().update_acquisition_function(function, models, datasets)
        #return self._expected_improvement_fn if acquistion_function == self._constrained_improvement_fn else acquistion_function
        tf.debugging.Assert(self._expected_improvement_fn is not None, [tf.constant([])])
        return self._expected_improvement_fn


# %% [markdown]
# ## Run

# %%
class Run:
    class Results:
        class Result:
            def __init__(self, constraints, global_x_max, global_f_max, feas_x_max, feas_f_max, data, time):
                self.global_x_max = global_x_max
                self.global_f_max = global_f_max
                self.feas_x_max = feas_x_max
                self.feas_f_max = feas_f_max
                self.sat = np.squeeze(constraints_satisfied(constraints, data[0]))
                self.res = np.squeeze(constraints_residual(constraints, data[0]))
                self.res_fail = np.where(self.res<-ctol, self.res, 0.)
                self.points = np.squeeze(data[0])
                self.fs = np.squeeze(data[1])

                # Allow for feas_f_max to be inaccurate. Optimizer can find a better solution.
                if self.fs <= self.feas_f_max:
                    self.xerr = np.linalg.norm(self.points-self.feas_x_max)
                    self.ferr = np.linalg.norm(self.fs-self.feas_f_max)
                else:
                    self.xerr = np.float64(0.)
                    self.ferr = np.float64(0.)

                self.cerr = np.linalg.norm(self.res_fail)
                self.time = time
    
            def __str__(self):
                precision = 3
                with np.printoptions(precision=3):
                    out = "\nGlobal max"
                    out += "\n\tx\t\t\t:" + np.array2string(self.global_x_max)
                    out += "\n\tf\t\t\t:" + np.array2string(self.global_f_max)
                    out += "\nFeasible max"
                    out += "\n\tx\t\t\t:" + np.array2string(self.feas_x_max)
                    out += "\n\tf\t\t\t:" + np.array2string(self.feas_f_max)
                    out += "\nOptimization"
                    out += "\n\ttime\t\t\t:" + f"{self.time:.3f}s"
                    out += "\n\tx\t\t\t:" + np.array2string(self.points, precision=precision)
                    out += "\n\tf\t\t\t:" + np.array2string(self.fs, precision=precision)
                    out += "\n\txerr\t\t\t:" + np.array2string(self.xerr, precision=precision)
                    out += "\n\tferr\t\t\t:" + np.array2string(self.ferr, precision=precision)
                    out += "\n\tconstr-satisfied\t:" + str(self.sat)
                    if not self.sat:
                        out += "\n\t\tcerr\t\t:" + np.array2string(self.cerr, precision=precision)
                return textwrap.indent(out, '\t\t')
    
        def __init__(self, label):
            self.label = label
            self._results = []
    
        def evaluate(self, constraints, global_x_max, global_f_max, feas_x_max, feas_f_max, data, time):
            self._results.append(Run.Results.Result(constraints, global_x_max, global_f_max, feas_x_max, feas_f_max, data, time))
            
        def __len__(self):
            return len(self._results)
        
        @property
        def results(self):
            return self._results
        
        def __str__(self):
            out = self.label
            for i, r in enumerate(self._results):
                out += f"\n\t{i}:"
                out += str(r)
            return out

    def __init__(self, search_space, observer, constraints, constrained_ei_type=None, builder_kwargs=None):
        self.search_space = search_space
        self.observer = observer
        self.constraints = constraints
        self.optims = {}
        self.data = {}
        self.results_hist = {}
        self.plot_frames = []
        self.constrained_ei_type = constrained_ei_type
        self.builder_kwargs = builder_kwargs

        if len(self.search_space.lower) == 2:
            # Create a grid in 2D for plotting.
            [self.Xi, self.Xj] = np.meshgrid(np.linspace(self.search_space.lower[0], self.search_space.upper[0], 50), np.linspace(self.search_space.lower[1], self.search_space.upper[1], 50))
            self.X = np.vstack((self.Xi.ravel(), self.Xj.ravel())).T  # Change our input grid to list of coordinates.
            self.C = constraints_satisfied(self.constraints, self.X).astype(int)
        else:
            # For non-2D, use search_space sampling.
            [self.Xi, self.Xj] = None, None
            self.X = None
            self.C = None
            #self.X = self.search_space.sample_sobol(10**len(self.search_space.lower)).numpy()
            #self.C = constraints_satisfied(self.constraints, self.X).astype(int)

    def get_acq_fun(self):
        if self.constrained_ei_type is not None:
            feas = SimpleConstraintsFeasibility(self.constraints)
            default_builder_kwargs = dict(min_feasibility_probability=0.5)
            acq_builder = self.constrained_ei_type(OBJECTIVE, feas.using(OBJECTIVE), **dict(default_builder_kwargs, **dict(self.builder_kwargs or {})))
            acq_function = acq_builder.prepare_acquisition_function({OBJECTIVE: self.model}, datasets={OBJECTIVE: self.initial_data})
        else:
            acq_builder = ExpectedImprovement(**dict(self.builder_kwargs or {}))
            acq_function = acq_builder.prepare_acquisition_function(self.model, dataset=self.initial_data)

        return acq_builder, acq_function
        
    def build_fn(self, initial_query_points):
        self.initial_data = self.observer(initial_query_points)

        gpflow_model = build_gpr(self.initial_data, self.search_space, likelihood_variance=1e-7)
        self.model = GaussianProcessRegression(gpflow_model)

        # Create and call acquisition function so the tensorflow graph is built upfront, so graph compile time is excluded from measurement.
        # Creates reference values.
        self.acq_builder, self.acq_function = self.get_acq_fun()

        if self.X is not None:
            self.acq_f_vals = np.squeeze(self.acq_function(tf.expand_dims(self.X, axis=-2)))

            global_ind_max = np.argmax(self.acq_f_vals)
            self.global_f_max = self.acq_f_vals[global_ind_max]
            self.global_x_max = self.X[global_ind_max]

            f_masked = np.where(self.C, self.acq_f_vals, np.NINF)
            feas_ind_max = np.argmax(f_masked)
            self.feas_f_max = f_masked[feas_ind_max]
            self.feas_x_max = self.X[feas_ind_max]
        else:
            self.acq_f_vals = None
            self.global_f_max = np.NINF
            self.global_x_max = None
            self.feas_f_max = np.NINF
            self.feas_x_max = None

            # Random sampling in batches to keep low memory footprint.
            for _ in range(10):
                X = self.search_space.sample(10**len(self.search_space.lower)).numpy()
                C = constraints_satisfied(self.constraints, X).astype(int)

                acq_f_vals = np.squeeze(self.acq_function(tf.expand_dims(X, axis=-2)))
                global_ind_max = np.argmax(acq_f_vals)
                global_f_max = acq_f_vals[global_ind_max]
                global_x_max = X[global_ind_max]
                if global_f_max > self.global_f_max:
                    self.global_f_max = global_f_max
                    self.global_x_max = global_x_max

                f_masked = np.where(C, acq_f_vals, np.NINF)
                feas_ind_max = np.argmax(f_masked)
                feas_f_max = f_masked[feas_ind_max]
                feas_x_max = X[feas_ind_max]
                if feas_f_max > self.feas_f_max:
                    self.feas_f_max = feas_f_max
                    self.feas_x_max = feas_x_max
        
    def add_optims(self, optims):
        self.optims.update(optims)
        
    def run_optims(self, num_initial_samples, num_optimization_runs, init_candidates=None):
        for name, optim in self.optims.items():
            try:
                data, optim_timer = self._run_optim(num_initial_samples, num_optimization_runs, init_candidates, optim)
            except BaseException as err:
                print(f"Optimisation for {name} failed: {err=}, {type(err)=}")
            else:
                self.data[name] = data
                if name not in self.results_hist:
                    self.results_hist[name] = Run.Results(name)
                self.results_hist[name].evaluate(self.constraints, self.global_x_max, self.global_f_max, self.feas_x_max, self.feas_f_max, self.data[name], optim_timer.time)
        
    def _run_optim(self, num_initial_samples, num_optimization_runs, init_candidates, optimizer_args):
        init_points = np.zeros((num_optimization_runs, 1, len(self.search_space.lower)))
        opt = generate_continuous_optimizer(
            num_initial_samples=num_initial_samples,
            num_optimization_runs=num_optimization_runs,
            initial_candidates_in=init_candidates,
            initial_points_out=init_points,
            optimizer_args=optimizer_args,
        )
        init_points = init_points.squeeze()

        acq_rule = EfficientGlobalOptimization(  # type: ignore
            num_query_points=1, builder=self.acq_builder, optimizer=opt, initial_acquisition_function=self.acq_function
        )
        
        with Timer() as optim_timer:
            points_chosen = acq_rule.acquire_single(self.search_space, self.model, dataset=self.initial_data).numpy()
        
        f_val = self.acq_function(tf.expand_dims(points_chosen, axis=-2))

        return (points_chosen, f_val, init_points), optim_timer

    def print_results_full(self):
        for result in self.results_hist.values():
            print(result)

    def print_results_summary(self):
        for result in self.results_hist.values():
            sats = np.array([r.sat for r in result.results])
            xerrs = np.array([r.xerr for r in result.results])
            ferrs = np.array([r.ferr for r in result.results])
            cerrs = np.array([r.cerr for r in result.results])
            ts    = np.array([r.time for r in result.results])
            print(f"{result.label}\t(runs={len(result)}, mean-secs={np.mean(ts):.3f}): constr-sat-%={np.mean(sats)*100:5.1f}, mean-xerr={np.mean(xerrs):.3f}," +
                  f" mean-ferr={np.mean(ferrs):.3f}, mean-cerr={np.mean(cerrs):.3f}")

    def _plot_optim(self, ax, name=None):
        if name is not None:
            points, fs, init_points = self.data[name]
            fs = np.squeeze(fs, axis=-1)
        else:
            points, fs, init_points = None, None, None

        plot_obj = ax.contour(self.Xi, self.Xj, np.reshape(self.acq_f_vals, self.Xi.shape), levels=80, zorder=0)
        ax.contourf(self.Xi, self.Xj, np.reshape(self.C, self.Xi.shape), levels=1, colors=[(.2, .2, .2, .7), (1, 1, 1, 0)], zorder=1)
        
        if init_points is not None:
            ax.scatter(
                init_points[:, 0],
                init_points[:, 1],
                color="magenta",
                lw=1,
                label="Initial points",
                marker="o",
            )
        
        if points is not None:
            ax.scatter(
                points[:, 0],
                points[:, 1],
                color="red",
                lw=5,
                label=name,
                marker="*",
            )
            ax.annotate(
                #[f"{f:.3}" for f in fs], 
                f"{float(fs):.3}",
                (points[:, 0], points[:, 1]),
                textcoords="offset points",
                xytext=(10,-15),
                bbox=dict(boxstyle="round", fc="w", alpha=0.7),
            )
        
        ax.scatter(
            self.global_x_max[0],
            self.global_x_max[1],
            color="blue",
            lw=2,
            label="Global max",
            marker="^",
        )
        ax.annotate(
            f"{self.global_f_max:.3}",
            (self.global_x_max[0], self.global_x_max[1]),
            textcoords="offset points",
            xytext=(5,10),
            bbox=dict(boxstyle="round", fc="w", alpha=0.7),
        )
        
        ax.scatter(
            self.feas_x_max[0],
            self.feas_x_max[1],
            color="black",
            lw=4,
            label="Feasible max",
            marker="+",
        )
        ax.annotate(
            f"{self.feas_f_max:.3}", 
            (self.feas_x_max[0], self.feas_x_max[1]),
            textcoords="offset points",
            xytext=(5,10),
            bbox=dict(boxstyle="round", fc="w", alpha=0.7),
        )
    
        ax.legend(bbox_to_anchor=(1.25, 1), loc="upper left")
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        cbar = plt.colorbar(plot_obj)
        cbar.set_label("EI", rotation=270, labelpad=10)

    def plot_results(self):
        num_plots = len(self.results_hist)
        num_per_side = int(np.ceil(num_plots/2))
        fig, ax = plt.subplots(num_per_side, num_per_side, figsize=(7.2*num_per_side, 4.8*num_per_side))

        for _ in range(len(np.shape(ax)), 2):
            ax = np.expand_dims(ax, axis=-1)

        for i, name in enumerate(self.results_hist.keys()):
            self._plot_optim(ax[i//num_per_side, i%num_per_side], name)

        fig.tight_layout(pad=0.8)
        plt.savefig('figure.png')

        fig.canvas.draw()
        size_pix = fig.get_size_inches()*fig.dpi
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        self.plot_frames.append(image.reshape(list(size_pix[::-1].astype(int)) + [3]))

    def write_gif(self):
        imageio.mimsave('figures.gif', self.plot_frames, format='GIF', fps=.20)


# %%
def multi_run(run, num_models, num_optims, initial_query_points, num_initial_samples=2, num_optimization_runs=2, with_plot=True):
    plt_be = plt.get_backend()
    plt.switch_backend('Agg')
    
    progress_widget = ipywidgets.FloatProgress(value=0., min=0., max=1.)
    run_status_widget = ipywidgets.Label(align='right')
    hbox_widget = ipywidgets.HBox([progress_widget, ipywidgets.Label("run: "), run_status_widget])

    def display_status(progress: float, run_status: str = "") -> None:
        progress_widget.value = progress
        run_status_widget.value = run_status

    # display() only exists when run under jupyter notebook/lab.
    try:
        display(hbox_widget)
    except BaseException:
        pass

    total_runs = num_models * num_optims
    plot_period = max(total_runs/5, 1)
    for i in range(num_models):
        run.build_fn(initial_query_points())
        
        #num_initial_samples = tf.maximum(NUM_SAMPLES_MIN, NUM_SAMPLES_DIM * tf.shape(search_space.lower)[-1])
        #num_optimization_runs = NUM_RUNS_DIM * tf.shape(search_space.lower)[-1]
        
        #test_initial_candidates = search_space.sample(num_initial_samples)  # In case the number of hard-coded candidates is less than required number, fill with samples.
        #test_initial_candidates = np.array([[0.1, 0.1], [0.8, 0.2]])
        #test_initial_candidates = np.expand_dims(test_initial_candidates, axis=1)
        #
        #display_status(i*num_optims/total_runs, f"{i*num_optims+1}/{total_runs}")
        #run.run_optims(num_initial_samples, num_optimization_runs, test_initial_candidates)
        #if (i*num_optims)%plot_period == 0:
        #    run.plot_results()
        
        for j in range(0, num_optims):
            display_status((i*num_optims+j)/total_runs, f"{i*num_optims+j+1}/{total_runs}")
            run.run_optims(num_initial_samples, num_optimization_runs)
            if with_plot and (i*num_optims+j)%plot_period == 0:
                run.plot_results()

    display_status(1., "done")

    plt.switch_backend(plt_be)

# %% [markdown]
# ## LICENSE
#
# [Apache License 2.0](https://github.com/secondmind-labs/trieste/blob/develop/LICENSE)
