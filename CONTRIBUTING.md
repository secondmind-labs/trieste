# Contribution guidelines

### Community guidelines

Community is important to us, and we want everyone to feel welcome and be able to contribute to their fullest. Our [code of conduct](CODE_OF_CONDUCT.md) gives an overview of what that means.


### Reporting a bug

Finding and fixing bugs helps us provide robust functionality to all users. If you think you've found a bug, we encourage you either to submit a bug report or, if you know how to fix the bug yourself, submit a fix. If you would like help with either, ask in the #trieste channel of Secondmind Labs' [community Slack workspace](https://join.slack.com/t/secondmind-labs/shared_invite/zt-ph07nuie-gMlkle__tjvXBay4FNSLkw).

We use GitHub issues for bug reports. You can use the [bug issue template](https://github.com/secondmind-labs/trieste/issues/new?assignees=&labels=bug&template=bug_report.md&title=) to start writing yours. Once you've submitted it, the maintainers will take a look as soon as possible, ideally within the week, and get back to you about how to proceed. If it's a small easy fix, they may implement it then and there. For fixes that are more involved, they may discuss with you about how urgent the fix is, with the aim of providing some timeline of when you can expect to see it. If you haven't had a response after a week, feel free to comment again on the issue: notifications sometimes get lost!

If you'd like to submit a bug fix, please read the [Pull Request guidelines](#pull-request-guidelines) below and use the [pull request template](https://github.com/secondmind-labs/trieste/compare). We recommend you discuss your changes with the community before you begin working on them (either on Slack or in a GitHub issue) so that questions and suggestions can be made early on.


### Requesting a feature

Trieste is built on features added and improved by the community. Like with bugs, you can submit a feature request as an issue, or implement it yourself as a pull request. We gladly welcome either, but a Pull Request is likely to be released sooner, simply because others may not have time to implement it themselves.

When raising an issue, please use the [feature issue template](https://github.com/secondmind-labs/trieste/issues/new?assignees=&labels=&template=feature_request.md&title=). If you're interested in implementing it yourself, we recommend discussing it first with the community, either on GitHub or in the #trieste channel of Secondmind Labs' [community Slack workspace](https://join.slack.com/t/secondmind-labs/shared_invite/zt-ph07nuie-gMlkle__tjvXBay4FNSLkw). Among other things, this will help determine whether the feature lies within the scope of Trieste, as well as the best way to implement it while preserving backwards compatibility. If we feel the feature is not in scope, we may suggest adding it as a notebook or external extension instead.

Similar to bug reports, we will try to get back to you within the week, but do feel free to comment again if you think we may have not noticed!

### Code setup

To contribute code, ensure you install Trieste in editable mode by running the commands below from the root directory of the `trieste` repository:
```bash
git clone https://github.com/secondmind-labs/trieste.git
cd trieste
pip install -e .
```

If you wish to run the tests and format checks within your virtual environment or IDE, also install the following:

```bash
pip install -r common_build/taskipy/requirements.txt -c common_build/taskipy/constraints.txt
pip install -r common_build/format/requirements.txt -c common_build/format/constraints.txt
pip install -r common_build/types/requirements.txt -c common_build/types/constraints.txt
pip install -r tests/latest/requirements.txt -c tests/latest/constraints.txt
```

Alternatively, you can run the tests using `tox`, which is typically slower but more resilient, as it corresponds more precisely to way tests are run on GitHub before pull requests can be merged. To enable `tox`, ensure that it is installed:

````bash
pip install tox
````

To run the notebooks, there are additional setup instructions described [here](docs/README.md).

### Quality checks

[GitHub actions](https://docs.github.com/en/actions) will automatically run a number of quality checks against all pull requests to the `develop` branch. The GitHub repository is set up such that these need to pass in order to merge. Below is a summary of the various checks. We recommend manually running them before making a pull request, though it is also possible to wait until they are run on GitHub and fixing any errors in subsequent branch commits.

#### Type checking

We use [type hints](https://docs.python.org/3/library/typing.html) for documentation and static type checking with [mypy](http://mypy-lang.org). We do this throughout the source code and tests. The notebooks are checked for type correctness, but we only add types there if they are required for mypy to pass. This is because we anticipate most readers of notebooks won't be using type hints themselves. If you don't know how to add type hints when making a pull request, ask for help from reviewers or on the [community Slack workspace](./README.md#getting-help).

Run the type checker with
```bash
$ tox -e types
```

or, if running inside a virtual environment
```bash
task mypy
```

You can also pass in a path argument to type check just a single file or directory (if using tox, precede it with a ` -- ` separator).

#### Code formatting

We format all Python code with [black](https://black.readthedocs.io/en/stable/), [flake8](https://flake8.pycqa.org/en/latest/), and [isort](https://pycqa.github.io/isort/). To check whether your code satisfies the correct formatting requirements, run:

```bash
tox -e format
```

or, if you're using a virtual environment:

```bash
task check_format
```

To automatically apply all the formatting suggestions from black and isort, run:

```bash
tox -e reformat
```

or 

```bash
task format
```

Note that these formatting changes are affected by which version of black you're using. If you find that calling `task format` changes lots of files that you haven't touched, then you probably have the wrong version installed. Either install the correct one using the `pip install` commands described above, or use `tox` (which automatically installs the correct version).

#### Tests

We write and run tests with [pytest](https://pytest.org). We aim for all public-facing functionality to have tests for both happy and unhappy paths (that test it works as intended when used as intended, and fails as intended otherwise). We don't test private functionality, as the cost to ease of development is more problematic than the benefit of improved robustness.

You can run tests with
```bash
$ tox -e tests
```
or
```bash
task tests
```
To save time, some slower tests are not run by default. If you need to run them (either because they are clearly relevant to your code changes, or because you have been told that your code changes resulted in a failure) you can do so with:
```bash
$ tox -e tests -- --runslow yes
```
or
```bash
task tests --runslow yes
```

Running all of the tests (even without `--runslow`) takes a long time, and can result in out-of-memory issues. When developing your code it is usually enough to run locally just the tests or test directories that you're likely to have affected, and rely on GitHub to tell you whether you've broken any other tests. You can do this by passing in the test path to pytest:

```bash
$ tox -e tests -- <path>
```
or
```bash
task tests <path>
```


### External dependencies

To update the Python dependencies used in any part of the project (e.g. introducing a new external library), update setup.py and/or any relevant requirements.txt files. Then, in the repository root, and with all virtual environments deactivated, run
```bash
$ ./generate_constraints.sh
```
This will update all the constraints.txt files, but not your virtual environment. To update that, follow the `pip install` section of the [code setup](#code-setup) instructions.


### Documentation

Trieste has two primary sources of documentation: the [notebooks](https://secondmind-labs.github.io/trieste/develop/tutorials.html) and the [API reference](https://secondmind-labs.github.io/trieste/develop/autoapi/trieste/index.html).

For the API reference, we document Python code inline, using [reST markup](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html). See [here](docs/README.md) for details on the documentation build. All parts of the public API need docstrings (indeed anything without docstrings won't appear in the built documentation). Similarly, don't add docstrings to private functionality, otherwise it will appear in the documentation website. Use code comments sparingly, as they incur a maintenance cost and tend to drift out of sync with the corresponding code.

Significant new features should be documented in notebooks (which are also a useful tool while developing those features). If you end up working on such a feature, we recommend discussing this with the community.

### Pull request guidelines

When submitting Pull Requests, please:

- Limit the pull request to the smallest useful feature or enhancement, or the smallest change required to fix a bug. This makes it easier for reviewers to understand why each change was made, and makes reviews quicker.
- Include appropriate [documentation](#documentation) and [type hints](#type-checking), and ensure that all existing [tests](#tests) and [checks](#code-formatting) are passing.
- Include new tests. In particular:
  - New features should include a demonstration of how to use the new API, and should include sufficient tests to give confidence that the feature works as expected.
  - Bug fixes should include tests to verify that the updated code works as expected and defend against future regressions.
  - When refactoring code, please verify that existing tests are adequate.
- So that notebook users have the option to import things as
  ```python
  import trieste
  trieste.utils.objectives.branin(...)
  ```
  import all modules (or their contents) in their parent package `__init__.py` file.
- In commit messages, be descriptive but to the point. Comments such as "further fixes" obscure the more useful information.


# License

[Apache License 2.0](LICENSE)
