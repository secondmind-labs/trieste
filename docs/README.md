# Building documentation

To build the documentation, ensure you have both [tox](https://tox.readthedocs.io) and [pandoc](https://github.com/jgm/pandoc/releases/) installed.
Then run the following from the repository root directory:

```
$ tox -e docs
```

Open `docs/_build/html/index.html` in a browser to view the docs.

### Fixing documentation build errors

Any errors in the notebooks or API documentation markup will result in documentation
build errors that must be fixed. To fix these, ensure that all new documentation is valid [reST markup](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html).

Note that since the API documentation is generated from inline Python docstrings, 
**the line numbers in any API error logs refer to generated reST files,
not the original Python source files**. You can find these generated reST files
at `docs/autoapi`.

# License

[Apache License 2.0](../LICENSE)
