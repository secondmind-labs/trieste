# Trieste

[![PyPI](https://img.shields.io/pypi/v/trieste.svg)](https://pypi.org/project/trieste)
[![License](https://img.shields.io/badge/license-Apache-green.svg)](LICENSE)
[![Quality checks](https://github.com/secondmind-labs/trieste/actions/workflows/quality-checks.yaml/badge.svg)](https://github.com/secondmind-labs/trieste/actions?query=workflows%3Aquality-checks)
[![Docs](https://github.com/secondmind-labs/trieste/actions/workflows/deploy.yaml/badge.svg)](https://github.com/secondmind-labs/trieste/actions/workflows/deploy.yaml)
[![Codecov](https://img.shields.io/codecov/c/github/secondmind-labs/trieste/coverage.svg?branch=master)](https://codecov.io/github/secondmind-labs/trieste?branch=master)
[![Slack Status](https://img.shields.io/badge/slack-trieste-green.svg?logo=Slack)](https://join.slack.com/t/secondmind-labs/shared_invite/zt-ph07nuie-gMlkle__tjvXBay4FNSLkw)


[Documentation (release)](https://secondmind-labs.github.io/trieste/1.0.0/index.html) |
[Documentation (develop)](https://secondmind-labs.github.io/trieste/develop/index.html) |
[Tutorials](https://secondmind-labs.github.io/trieste/1.0.0/tutorials.html) |
[API reference](https://secondmind-labs.github.io/trieste/1.0.0/autoapi/trieste/index.html) |


## What does Trieste do?

Trieste (pronounced tree-est) is a Bayesian optimization toolbox built on [TensorFlow](https://www.tensorflow.org/). Trieste is named after the bathyscaphe Trieste, the first vehicle to take a crew to Challenger Deep in the Mariana Trench, the lowest point on the Earth’s surface: the literal global minimum.


## Getting started




In the [Documentation](https://secondmind-labs.github.io/trieste/), we have multiple [Tutorials](https://secondmind-labs.github.io/trieste/tutorials.html) showing the basic functionality of the toolbox, a [benchmark implementation](https://secondmind-labs.github.io/trieste/notebooks/benchmarks.html) and a comprehensive [API reference](https://secondmind-labs.github.io/trieste/autoapi/gpflux/index.html).


Trieste has a [documentation site](https://secondmind-labs.github.io/trieste) with tutorials on how to use the library, and an API reference. You can also run the tutorials interactively. They can be found in the notebooks directory, and are written as Python scripts for running with Jupytext. To run them, first install trieste from sources as above, then install additional 


## Installation

Trieste supports Python 3.7 onwards and uses [semantic versioning](https://semver.org/).


#### For users

To install the latest (stable) release of the toolbox from [PyPI](https://pypi.org/), use `pip`:
```bash
$ pip install trieste
```
or to install from sources, run
```bash
$ pip install .
```
in the repository root.


#### For contributors

To install this project in editable mode, run the commands below from the root directory of the `trieste` repository.
```bash
pip install -e .
```
For detailed instructions, see [the guidelines for contributors](CONTRIBUTING.md).


#### Tutorials

Trieste has a [documentation site](https://secondmind-labs.github.io/trieste) with tutorials on how to use the library, and an API reference. You can also run the tutorials interactively. They can be found in the notebooks directory, and are written as Python scripts for running with Jupytext. To run them, first install trieste from sources as above, then install additional dependencies with
```bash
$ pip install -r notebooks/requirements.txt
```
Finally, run the notebooks with
```bash
$ jupyter-notebook notebooks
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

To cite Trieste, please reference our [arXiv](https://arxiv.org/) paper where we review the framework and describe the design. Sample Bibtex is given below:

```
@article{trieste2023,
  author = {XXX},
    title = "{XXX}",
  journal = {XXX},
  year    = {2023},
  url     = {http://arxiv.org/XXX}
}
```

## License

[Apache License 2.0](LICENSE)
