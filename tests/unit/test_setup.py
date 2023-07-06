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

import pytest
import yaml

BASE_PATH = Path(__file__).parents[2]
VERSION = BASE_PATH / "trieste" / "VERSION"
VERSIONS = BASE_PATH / "versions.json"
CITATION = BASE_PATH / "CITATION.cff"
REDIRECT = BASE_PATH / "redirect.html"
REDIRECT_AUTOAPI = BASE_PATH / "redirect_autoapi.html"
REDIRECT_TUTORIALS = BASE_PATH / "redirect_tutorials.html"


@pytest.fixture(name="version")
def _version() -> str:
    print(__file__)
    return VERSION.read_text().strip()


@pytest.fixture(name="versions")
def _versions() -> list[dict[str, Any]]:
    with open(VERSIONS) as f:
        return json.load(f)


@pytest.fixture(name="citation")
def _citation() -> list[dict[str, Any]]:
    with open(CITATION) as f:
        return yaml.safe_load(f)


@pytest.fixture(name="redirect")
def _redirect() -> str:
    return REDIRECT.read_text()


@pytest.fixture(name="redirect_autoapi")
def _redirect_autoapi() -> str:
    return REDIRECT_AUTOAPI.read_text()


@pytest.fixture(name="redirect_tutorials")
def _redirect_tutorials() -> str:
    return REDIRECT_TUTORIALS.read_text()


def test_version_is_valid(version: str) -> None:
    assert re.match(r"\d+\.\d+\.\d+", version)


def test_versions_is_valid(versions: list[dict[str, Any]]) -> None:
    assert isinstance(versions, list)
    for v in versions:
        assert isinstance(v, dict)
        assert set(v.keys()) == {"version", "url"}
        assert all(isinstance(value, str) for value in v.values())
        assert v["url"] == f"https://secondmind-labs.github.io/trieste/{v['version']}/"


def test_version_in_versions(version: str, versions: list[dict[str, Any]]) -> None:
    assert any(v["version"] == version for v in versions)


def test_citation_version(version: str, citation: dict[str, Any]) -> None:
    assert citation["version"] == version


def test_redirect_version(
    version: str, redirect: str, redirect_autoapi: str, redirect_tutorials: str
) -> None:
    assert "$VERSION/index.html" in redirect
    assert "$VERSION/autoapi/trieste/index.html" in redirect_autoapi
    assert "$VERSION/tutorials.html" in redirect_tutorials
