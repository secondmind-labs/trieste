# Trieste

A Bayesian optimization toolbox built on [TensorFlow](https://www.tensorflow.org/). Trieste is named after the bathyscaphe _Trieste_, the first vehicle to take a crew to _Challenger Deep_ in the _Mariana Trench_, the lowest point on the Earth's surface: the literal global minimum.

Supports Python 3.7 onwards. We use [semantic versioning](https://semver.org/).

We welcome contributions. See [the guidelines](CONTRIBUTING.md) to get started.

### Installation

To install trieste, run
```bash
$ pip install trieste
```

### Documentation

Trieste has a [documentation site](https://secondmind-labs.github.io/trieste) with tutorials on how to use the library, and an API reference. You can also run the tutorials interactively. They can be found in the notebooks directory, and are written as Python scripts for running with Jupytext. To run them, first install the library as above, then additional dependencies with
```bash
$ pip install -r notebooks/requirements.txt -c notebooks/constraints.txt
```
Finally, run the notebooks with
```bash
$ jupyter-notebook notebooks
```
Note: If you experience missing dependencies, check the notebooks are using the same python environment in which the dependencies were installed.

### Getting help

- To submit a pull request, file a bug report, or make a feature request, see the [contribution guidelines](CONTRIBUTING.md).
- For more open-ended questions, or for anything else, join the community discussions on our [Slack workspace](https://join.slack.com/t/secondmind-labs/shared_invite/zt-hsl5m8oj-m5VW8_ND~~~GbxPXD1Dz4A).

# License

[Apache License 2.0](LICENSE)
