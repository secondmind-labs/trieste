# Trieste

A Bayesian optimization toolbox built on [TensorFlow](https://www.tensorflow.org/). Trieste supports Python 3.7 onwards and uses [semantic versioning](https://semver.org/).

We welcome contributions. See [the guidelines](CONTRIBUTING.md) to get started.

### Installation

To install trieste, run
```bash
$ pip install trieste
```
or to install from sources, run
```bash
$ pip install .
```
in the repository root.

### Documentation

Trieste has a [documentation site](https://secondmind-labs.github.io/trieste) with tutorials on how to use the library, and an API reference. You can also run the tutorials interactively. They can be found in the notebooks directory, and are written as Python scripts for running with Jupytext. To run them, first install trieste from sources as above, then install additional dependencies with
```bash
$ pip install -r notebooks/requirements.txt -c notebooks/constraints.txt
```
Finally, run the notebooks with
```bash
$ jupyter-notebook notebooks
```

# License

[Apache License 2.0](LICENSE)
