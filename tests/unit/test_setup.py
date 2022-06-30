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
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import jsonschema
import pytest
import yaml

VERSION = Path("../../VERSION")
VERSIONS = Path("../../versions.json")
CITATION = Path("../../CITATION.cff")


@pytest.fixture(name="version")
def _version() -> str:
    return VERSION.read_text().strip()


@pytest.fixture(name="versions")
def _versions() -> list[dict[str, Any]]:
    with open(VERSIONS) as f:
        return json.load(f)


@pytest.fixture(name="citation")
def _citation() -> list[dict[str, Any]]:
    with open(CITATION) as f:
        return yaml.safe_load(f)


def test_version_is_valid(version: str) -> None:
    assert re.match(r"\d+\.\d+\.\d+", version)


def test_versions_is_valid(versions: list[dict[str, Any]]) -> None:
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "required": ["version", "url"],
            "properties": {"version": {"type": "string"}, "url": {"type": "string"}},
        },
    }
    jsonschema.validate(versions, schema)
    for v in versions:
        assert v["url"] == f"../{v['version']}/index.html"


def test_version_in_versions(version: str, versions: list[dict[str, Any]]) -> None:
    assert any(v["version"] == version for v in versions)


def test_citation_version(version: str, citation: dict[str, Any]) -> None:
    assert citation["version"] == version
