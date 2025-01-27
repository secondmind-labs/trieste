# Copyright 2020 The Trieste Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash

set -e

VENV_DIR=$(mktemp -d -p $(pwd))

generate_for_env () {
  # In a temporary virtual environment, install the dependencies in directory $1, and optionally
  # the library dependencies. Freeze the dependencies to a constraints file in directory $1.
  #
  # $1: The base path of the requirements and constraints files
  # $2: If true, installs the library dependencies
  python3 -m venv $VENV_DIR/$1
  source $VENV_DIR/$1/bin/activate
  pip install --upgrade pip
  if [ "$2" = true ]; then
      pip install -e .[qhsri]
  fi
  pip install -r $1/requirements.txt
  pip freeze --exclude-editable trieste > $1/constraints.txt
  deactivate
}

generate_for_env docs false
generate_for_env common_build/format false
generate_for_env common_build/taskipy false
generate_for_env common_build/types false
generate_for_env notebooks true
generate_for_env tests/old true
generate_for_env tests/prod true
generate_for_env tests/latest true

rm -rf $VENV_DIR
