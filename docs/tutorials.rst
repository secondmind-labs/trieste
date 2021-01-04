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

Tutorials
=========

The following tutorials demonstrate how to use Trieste to solve various optimization problems, and detail the functionality available in Trieste.

Example optimization problems
-----------------------------

Introductory
^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   notebooks/expected_improvement
   notebooks/thompson_sampling

Advanced
^^^^^^^^

.. toctree::
   :maxdepth: 1

   notebooks/inequality_constraints
   notebooks/failure_ego

Trieste functionality in detail
-------------------------------

.. toctree::
   :maxdepth: 1

   Create a custom acquisition function <notebooks/failure_ego>
   Create a custom model optimization routine <notebooks/failure_ego>

Run the tutorials interactively
-------------------------------

The above tutorials are built from Jupytext notebooks in the notebooks directory of the repository. These notebooks can also be run interactively. To do so, install the library from sources, along with additional notebook dependencies with (in the repository root)

.. code::

   $ pip install . -r notebooks/requirements.txt

then run

.. code::

   $ jupyter-notebook notebooks
