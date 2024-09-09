# Trieste

[![PyPI](https://img.shields.io/pypi/v/trieste.svg)](https://pypi.org/project/trieste)
[![License](https://img.shields.io/badge/license-Apache-green.svg)](LICENSE)
[![Release](https://img.shields.io/github/actions/workflow/status/secondmind-labs/trieste/release-checks.yaml?logo=github&label=release%20checks)](https://github.com/secondmind-labs/trieste/actions/workflows/release-checks.yaml)
[![Develop](https://img.shields.io/github/actions/workflow/status/secondmind-labs/trieste/develop-checks.yaml?logo=github&label=develop%20checks)](https://github.com/secondmind-labs/trieste/actions/workflows/develop-checks.yaml)
[![Codecov](https://img.shields.io/codecov/c/github/secondmind-labs/trieste/coverage.svg?branch=develop)](https://app.codecov.io/github/secondmind-labs/trieste/tree/develop)
[![Slack Status](https://img.shields.io/badge/slack-trieste-green.svg?logo=Slack)](https://join.slack.com/t/secondmind-labs/shared_invite/zt-ph07nuie-gMlkle__tjvXBay4FNSLkw)


[Documentation (develop)](https://secondmind-labs.github.io/trieste/develop/index.html) |
[Documentation (release)](https://secondmind-labs.github.io/trieste) |
[Tutorials](https://secondmind-labs.github.io/trieste/tutorials.html) |
[API reference](https://secondmind-labs.github.io/trieste/autoapi.html) |


## What does Trieste do?

Trieste (pronounced tree-est) is a Bayesian optimization toolbox built on [TensorFlow](https://www.tensorflow.org/). Trieste is named after the bathyscaphe [Trieste](https://en.wikipedia.org/wiki/Trieste_%28bathyscaphe%29), the first vehicle to take a crew to Challenger Deep in the Mariana Trench, the lowest point on the Earth’s surface: the literal global minimum.

**Why Trieste?**  
- Highly modular design and easily customizable. Extend it with your custom model or acquisition functions. Ideal for practitioners that want to use it in their systems or for researchers wishing to implement their latest ideas.
- Seamless integration with TensorFlow. Leveraging fully its auto differentiation - no more writing of gradients for your acquisition functions!, and scalability capabilities via its support for highly parallelized modern hardware (e.g. GPUs).
- General purpose toolbox. Advanced algorithms covering all corners of Bayesian optimization and Active learning - batch, asynchronous, constraints, multi-fidelity, multi-objective - you name it, Trieste has it. 
- Versatile model support out-of-the-box. From gold-standard Gaussian processes (GPs; [GPflow](https://github.com/GPflow/GPflow)) to alternatives like sparse variational GPs, Deep GPs ([GPflux](https://github.com/secondmind-labs/GPflux)) or Deep Ensembles ([Keras](https://keras.io/)), that scale much better with the number of function evaluations.
- Real-world oriented. Our Ask-Tell interface allows users to apply Bayesian optimization across a range of non-standard real-world settings where control over black-box function is partial. Built on TensorFlow and with comprehensive testing Trieste is production-ready.


## Getting started

Here's a quick overview of the main components of a Bayesian optimization loop. For more details, see our <span style="font-variant:small-caps;">[Documentation](https://secondmind-labs.github.io/trieste)</span> where we have multiple <span style="font-variant:small-caps;">[Tutorials](https://secondmind-labs.github.io/trieste/tutorials.html)</span> covering both the basic functionalities of the toolbox, as well as more advanced usage.

Let's set up a synthetic black-box objective function we wish to minimize, for example, a popular Branin optimization function, and generate some initial data
```python
from trieste.objectives import Branin, mk_observer

observer = mk_observer(Branin.objective)

initial_query_points = Branin.search_space.sample(5)
initial_data = observer(initial_query_points)
```

First step is to create a probabilistic model of the objective function, for example a Gaussian Process model
```python
from trieste.models.gpflow import build_gpr, GaussianProcessRegression

gpflow_model = build_gpr(initial_data, Branin.search_space)
model = GaussianProcessRegression(gpflow_model)
```

Next ingredient is to choose an acquisition rule and acquisition function
```python
from trieste.acquisition import EfficientGlobalOptimization, ExpectedImprovement

acquisition_rule = EfficientGlobalOptimization(ExpectedImprovement())
```

Finally, we optimize the acquisition function using our model for a number of steps and we check the obtained minimum
```python
from trieste.bayesian_optimizer import BayesianOptimizer

bo = BayesianOptimizer(observer, Branin.search_space)
num_steps = 15
result = bo.optimize(num_steps, initial_data, model)
query_point, observation, arg_min_idx = result.try_get_optimal_point()
```


## Installation

Trieste supports Python 3.9+ and TensorFlow 2.5+, and uses [semantic versioning](https://semver.org/).


### For users

To install the latest (stable) release of the toolbox from [PyPI](https://pypi.org/), use `pip`:
```bash
$ pip install trieste
```
or to install from sources, run
```bash
$ pip install .
```
in the repository root.


### For contributors

To install this project in editable mode, run the commands below from the root directory of the `trieste` repository.
```bash
git clone https://github.com/secondmind-labs/trieste.git
cd trieste
pip install -e .
```
For installation to be able to run quality checks, as well as other details, see [the guidelines for contributors](CONTRIBUTING.md).


### Tutorials

Trieste has a [documentation site](https://secondmind-labs.github.io/trieste) with tutorials on how to use the library, and an API reference. You can also run the tutorials interactively. They can be found in the notebooks directory, and are written as Python scripts for running with Jupytext. To run them, first install trieste from sources as above, then install additional dependencies with
```bash
$ pip install -r notebooks/requirements.txt
```
Finally, run the notebooks with
```bash
$ jupyter-notebook notebooks
```

Alternatively, you can copy and paste the tutorials into fresh notebooks and avoid installing the library from source. To ensure you have the required plotting dependencies, simply run:
```bash
$ pip install trieste[plotting]
```

### Importing Keras

Like [tensorflow-probability](https://www.tensorflow.org/probability), Trieste currently uses Keras 2. When using Tensorflow versions 2.16 onwards (which default to Keras 3) this needs to be imported from `tf_keras` rather than `tf.keras`. Alternatively, for a shortcut that works with all versions of Tensorflow, you can write:
```python
from gpflow.keras import tf_keras
```

## The Trieste Community

### Getting help

**Bugs, feature requests, pain points, annoying design quirks, etc:**
Please use [GitHub issues](https://github.com/secondmind-labs/trieste/issues/) to flag up bugs/issues/pain points, suggest new features, and discuss anything else related to the use of Trieste that in some sense involves changing the Trieste code itself. We positively welcome comments or concerns about usability, and suggestions for changes at any level of design. We aim to respond to issues promptly, but if you believe we may have forgotten about an issue, please feel free to add another comment to remind us.


### Slack workspace

We have a public [Secondmind Labs slack workspace](https://secondmind-labs.slack.com/). Please use this [invite link](https://join.slack.com/t/secondmind-labs/shared_invite/zt-ph07nuie-gMlkle__tjvXBay4FNSLkw) and join the #trieste channel, whether you'd just like to ask short informal questions or want to be involved in the discussion and future development of Trieste.


### Contributing

All constructive input is very much welcome. For detailed information, see [the guidelines for contributors](CONTRIBUTING.md).


## Citing Trieste

To cite Trieste, please reference our [arXiv](https://arxiv.org/abs/2302.08436) paper where we review the framework and describe the design. Sample Bibtex is given below:

```
@misc{trieste2023,
  author = {Picheny, Victor and Berkeley, Joel and Moss, Henry B. and Stojic, Hrvoje and Granta, Uri and Ober, Sebastian W. and Artemev, Artem and Ghani, Khurram and Goodall, Alexander and Paleyes, Andrei and Vakili, Sattar and Pascual-Diaz, Sergio and Markou, Stratis and Qing, Jixiang and Loka, Nasrulloh R. B. S and Couckuyt, Ivo},
  title = {Trieste: Efficiently Exploring The Depths of Black-box Functions with TensorFlow},
  publisher = {arXiv},
  year = {2023},
  doi = {10.48550/ARXIV.2302.08436},
  url = {https://arxiv.org/abs/2302.08436}
}
```

## License

[Apache License 2.0](LICENSE)
