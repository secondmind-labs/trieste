.. Copyright 2021 The Trieste Contributors

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

Example optimization problems
-----------------------------

The following tutorials explore various optimization problems using Trieste.

.. toctree::
   :maxdepth: 1

   notebooks/expected_improvement
   notebooks/batch_optimization
   notebooks/thompson_sampling
   notebooks/inequality_constraints
   notebooks/failure_ego
   notebooks/multi_objective_ehvi
   notebooks/deep_gaussian_processes
   notebooks/active_learning
   notebooks/feasible_sets

   


Frequently asked questions
--------------------------

The following tutorials (or sections thereof) explain how to use and extend specific Trieste functionality.

* :doc:`How do I set up a basic Bayesian optimization routine?<notebooks/expected_improvement>`
* :doc:`How do I set up a batch Bayesian optimization routine?<notebooks/batch_optimization>`
* :doc:`How do I build models with configuration dictionaries?<notebooks/model_config>`
* :ref:`How do I make a custom acquisition function?<notebooks/failure_ego:Create a custom acquisition function>`
* :doc:`How do I recover a failed optimization loop?<notebooks/recovering_from_errors>`
* :doc:`How do I track and visualize an optimization loop in realtime using TensorBoard?<notebooks/visualizing_with_tensorboard>`
* :doc:`Does Trieste have interface for external control of the optimization loop, also known as Ask-Tell interface?<notebooks/ask_tell_optimization>`
* :doc:`How do I perform data transformations required for training the model?<notebooks/data_transformation>`
* How do I use Trieste in asynchronous objective evaluation mode?

  * :doc:`Example of using greedy batch acquisition functions and Python multiprocessing module.<notebooks/asynchronous_greedy_multiprocessing>`
  * :doc:`Example of using non-greedy batch acquisition functions and Ray.<notebooks/asynchronous_nongreedy_batch_ray>`

.. toctree::
   :hidden:
   :maxdepth: 1

   notebooks/ask_tell_optimization
   notebooks/data_transformation
   notebooks/model_config
   notebooks/recovering_from_errors
   notebooks/asynchronous_greedy_multiprocessing
   notebooks/asynchronous_nongreedy_batch_ray
   notebooks/visualizing_with_tensorboard

Run the tutorials interactively
-------------------------------

The above tutorials are built from Jupytext notebooks in the notebooks directory of the repository. These notebooks can also be run interactively. To do so, install the library from sources, along with additional notebook dependencies, with (in the repository root)

.. code::

   $ pip install . -r notebooks/requirements.txt

then run

.. code::

   $ jupyter-notebook notebooks
