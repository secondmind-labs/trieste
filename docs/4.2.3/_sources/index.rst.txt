.. Copyright 2020 The Trieste Contributors

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

.. trieste documentation master file, created by sphinx-quickstart.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Trieste documentation
=====================

Trieste is a research toolbox built on `TensorFlow <http://www.tensorflow.org/>`_, dedicated to Bayesian optimization, the process of finding the optimal values of an expensive, black-box objective function by employing probabilistic models over observations.

Without loss of generality, Trieste only supports minimizing the
objective function. In the simplest case of an objective function with one-dimensional real
output :math:`f: X \to \mathbb R`, this is

.. math:: \mathop{\mathrm{argmin}}_{x \in X} f(x) \qquad .

When the objective function has higher-dimensional output, we can still talk of finding the minima,
though the optimal values will form a `Pareto set <https://en.wikipedia.org/wiki/Pareto_front>`_ rather than a single point. Trieste provides
functionality for optimization of single-valued objective functions, and supports extension to the
higher-dimensional case. It also supports optimization over constrained spaces, learning the
constraints alongside the objective.

Trieste (pronounced *tree-est*) is named after the bathyscaphe `Trieste <https://en.wikipedia.org/wiki/Trieste_(bathyscaphe)>`_, the first vehicle to take a crew to Challenger Deep in the Mariana Trench, the lowest point on the Earth's surface: the literal global minimum.

Installation
------------

To install Trieste, run

.. code::

   $ pip install trieste

The library supports Python 3.9 onwards, and uses `semantic versioning <https://semver.org/>`_.

Getting help
------------

- We welcome contributions. To submit a pull request, file a bug report, or make a feature request, see the `contribution guidelines <https://github.com/secondmind-labs/trieste/blob/develop/CONTRIBUTING.md>`_.
- For more open-ended questions, or for anything else, join the discussions on Trieste channels in Secondmind Labs' community `Slack workspace <https://join.slack.com/t/secondmind-labs/shared_invite/zt-ph07nuie-gMlkle__tjvXBay4FNSLkw>`_.

.. toctree::
   :hidden:

   Trieste <self>

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :hidden:

   API reference <autoapi/trieste/index>

.. toctree::
   :maxdepth: 1
   :hidden:

   Tutorials <tutorials>

.. toctree::
   :maxdepth: 1
   :hidden:

   Bibliography <bibliography>
