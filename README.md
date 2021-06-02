# Code for Scalable Thompson Sampling using Sparse Gaussian Process Models

This codebase contains code to run the S-GP-TS algorithm (decoupled Thompson sampling from an sparse variational Gaussian process). 

This codebase is a fork of Trieste : a Bayesian optimization toolbox built on [TensorFlow](https://www.tensorflow.org/). 

### Documentation

We include a notebook S_GP_TS_demo.pct.py (in ./docs/notebooks) that allows the running of S-GP-TS on synthetic benchmark problems.

The notebook can be found in the notebooks directory, and is written as a Python script for running with Jupytext. To run it, first install this Trieste fork with

```bash
$ pip install .
```
then install additional dependencies with
```bash
$ pip install -r notebooks/requirements.txt
```
Finally, run the notebooks with
```bash
$ jupyter-notebook notebooks
```

For more general guidelines about Trieste, we recommend Trieste's  [documentation site](https://secondmind-labs.github.io/trieste) with tutorials on how to use the library, and an API reference. 