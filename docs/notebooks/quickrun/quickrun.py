# Copyright 2021 The Trieste Contributors
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

"""
A script to apply modifications to the notebook scripts based on YAML config,
used to make them run more quickly in continuous integration.
"""
from jsonschema import validate
from pathlib import Path
import re
import sys
import yaml
import logging
import argparse

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# The YAML config files specify a sequence of line replacements to apply.
# The "from" field is a regular expression that must match the entire line (apart from leading
# whitespace).
YAML_CONFIG_SCHEMA = """
type: object
properties:
  replace:
    type: array
    items:
      type: object
      properties:
        from:
          type: string
        to:
          type: string
      required:
        - from
        - to 
required:
  - replace
"""

JSON_SCHEMA = yaml.safe_load(YAML_CONFIG_SCHEMA)


def modify_all(revert: bool = False) -> None:
    """
    Modify all the notebooks that have corresponding YAML config to/from quickrun.

    :param revert: whether to revert the modifications
    :raise ValidationError: if any of the YAML configs are invalid
    """
    base = Path(sys.path[0])
    for path in sorted(base.glob("../*.pct.py")):
        config_path = Path(base / path.name.replace(".pct.py", ".yaml"))
        if config_path.exists():
            with config_path.open(encoding="utf-8") as fp:
                config = yaml.safe_load(fp)
            validate(config, JSON_SCHEMA)
            revert_notebook(path) if revert else modify_notebook(path, config)
        else:
            logger.info("No YAML config found for %s", path.name)


def modify_notebook(path: Path, config: dict) -> None:
    """
    Modify a notebook using the given config.

    :param path: notebook path
    :param config: loaded config
    :raise ValueError: If the config specifies a substitution that doesn't match the notebook.
    """
    notebook = path.read_text(encoding="utf-8")
    if "# quickrun" in notebook:
        logger.warning("Already modified %s for quickrun", path.name)
        return

    for repl in config["replace"]:
        repl_from = "^(( *){})$".format(repl["from"])
        repl_to = "\\2{} # quickrun \\1".format(repl["to"])
        if not re.search(repl_from, notebook, flags=re.MULTILINE):
            raise ValueError("Quickrun replacement %r doesn't match file %s" % (repl["from"], path))
        notebook = re.sub(repl_from, repl_to, notebook, flags=re.MULTILINE)
    path.write_text(notebook, encoding="utf-8")
    logger.info("Modified %s", path.name)


def revert_notebook(path: Path) -> None:
    """
    Revert a notebook from quickrun format.

    :param path: notebook path
    :param config: loaded config
    """

    notebook = path.read_text(encoding="utf-8")
    if "# quickrun" not in notebook:
        logger.info("No need to revert %s", path.name)
        return

    notebook = re.sub("^.* # quickrun (.*)$", "\\1", notebook, flags=re.MULTILINE)
    path.write_text(notebook, encoding="utf-8")
    logger.info("Reverted %s", path.name)


def main() -> None:
    """Script entry point."""
    parser = argparse.ArgumentParser(
        description="Modify notebook files for continuous integration."
    )
    parser.add_argument("--revert", action="store_true", help="Revert notebook files")
    args = parser.parse_args()
    modify_all(revert=args.revert)


if __name__ == "__main__":
    print(Path(sys.path[0]))
    main()
