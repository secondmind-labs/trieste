# Copyright 2022 The Trieste Contributors
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
"""This module exposes the trieste version number."""
from pathlib import Path

BASE_PATH = Path(__file__).parents[0]
VERSION = BASE_PATH / "VERSION"

__version__ = Path(VERSION).read_text(encoding="utf8").strip()
