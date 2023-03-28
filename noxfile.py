# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Test automation using `nox`."""

from pathlib import Path
from typing import Dict, Final, List, Union

import nox
import yaml
from nox import Session
from nox.virtualenv import CondaEnv

PYTHON: Final = ["3.8", "3.9", "3.10"]
CI_ENV: Final = (
    "black[jupyter]",
    "isort",
    "flake8",
    "flake8-annotations",
    "flake8-black",
    "flake8-bugbear",
    "flake8-docstrings",
    "flake8-import-order",
    "darglint",
    "mypy",
    "types-pyyaml",
    "pytest",
    "pytest-benchmark",
    "pytest-cov",
)

nox.options.sessions = ["ci"]


def _setup(session: Session) -> None:
    """Install `av2` into a virtual environment.

    Args:
        session: `nox` session.
    """
    if isinstance(session.virtualenv, CondaEnv):
        # Load environment.yml file and install conda dependencies.
        env = yaml.safe_load(Path("conda/environment.yml").read_text())

        conda_pkgs: List[str] = []
        reqs: List[str] = []
        pkgs: List[Union[str, Dict[str, str]]] = env["dependencies"]
        for pkg in pkgs:
            if isinstance(pkg, dict):
                if "pip" in pkg:
                    reqs += pkg["pip"]
            else:
                conda_pkgs.append(pkg)
        session.conda_install(*conda_pkgs)

        # Install pip dependencies if they exist.
        if len(reqs) > 0:
            session.install(*reqs, "--no-deps")

    # Install package.
    session.install("-e", ".")


@nox.session(python=PYTHON)
def ci(session: Session) -> None:
    """Run CI against `av2`.

    Args:
        session: `nox` session.
    """
    _setup(session)
    session.install(*CI_ENV)
    session.run("black", ".")
    session.run("isort", ".")
    session.run("flake8", ".")
    session.run("mypy", ".")
    session.run("pytest", "tests", "--cov", "src/av2")
