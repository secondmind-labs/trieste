# Building documentation

To build the documentation, ensure you have both [tox](https://tox.readthedocs.io) and [pandoc](https://github.com/jgm/pandoc/releases/) installed.
Then run the following from the repository root directory:

```
$ tox -e docs
```

Open `docs/_build/html/index.html` in a browser to view the docs.

Alternatively, to build the docs in a virtual environment rather than using tox, first ensure you
have the necessary requirements installed:

```bash
$ pip install -r notebooks/requirements.txt -c notebooks/constraints.txt
$ pip install -r docs/requirements.txt -c docs/constraints.txt
```

And then run `make html` in the `docs` subdirectory.

### Fixing documentation build errors

Any errors in the notebooks or API documentation markup will result in documentation
build errors that must be fixed. To fix these, ensure that all new documentation is valid
[reST markup](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html).

Note that since the API documentation is generated from inline Python docstrings, 
**the line numbers in any API error logs refer to generated reST files,
not the original Python source files**. You can find these generated reST files
at `docs/autoapi`.

### Building just parts of the documentation

To save time, you can build just the API documentation (ignoring the notebooks) by
setting up a virtual environment as described above and running the following from the
`docs` subdirectory. Note that this will report warnings about the missing notebooks,
which you can ignore.

```bash
$ sphinx-build -M html . _build -D exclude_patterns=_build,Thumbs.db,.DS_Store,notebooks
```

The easiest way to test a specific notebook for *Python errors* is to run it with `python`
or in `jupyter-notebook`, as described in the [notebooks README](notebooks/README.md). However,
if you wish to build the documentation for a specific notebook, you can run something like
the following command to exclude all other notebooks and the API documentation:

```bash
$ sphinx-build -M html . _build -D autoapi_dirs= -D exclude_patterns=_build,Thumbs.db,.DS_Store,notebooks/[a-su-z]*
```

### Partial executions of the notebooks

For continuous integration, we save time by executing only dummy runs of the notebook 
optimization loops (the notebooks are still executed in full after merging).
To do this locally, you can run the following:

```bash
$ tox -e quickdocs
```

These partial runs are managed by the 
configuration in the `docs/notebooks/quickrun` subdirectory.
To run them in your own virtual environment, execute the following from the `docs`
subdirectory before building the documentation:
```bash
$ python notebooks/quickrun/quickrun.py
```
Note that this will modify the notebook files.
To revert them to what they were before, you can execute the following:
```bash
$ python notebooks/quickrun/quickrun.py --revert
```

# License

[Apache License 2.0](../LICENSE)
