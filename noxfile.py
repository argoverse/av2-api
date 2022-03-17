# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>

"""Test automation using `nox`."""

from pathlib import Path
from sys import platform
from typing import Dict, Final, List, Union

import nox
import yaml
from nox import Session
from nox.virtualenv import CondaEnv

PYTHON: Final[List[str]] = ["3.8", "3.9", "3.10"]

nox.options.sessions = ["black", "isort", "lint", "mypy", "pytest"]
if platform != "win32":  # pytype isn't supported on windows.
    nox.options.sessions.append("pytype")


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
def black(session: Session) -> None:
    """Run `black` against `av2`.

    Args:
        session: `nox` session.
    """
    env = ["black[jupyter]"]
    _setup(session)
    session.install(*env)
    session.run("black", ".")


@nox.session(python=PYTHON)
def isort(session: Session) -> None:
    """Run `isort` against `av2`.

    Args:
        session: `nox` session.
    """
    env = ["isort"]
    _setup(session)
    session.install(*env)
    session.run("isort", ".")


@nox.session(python=PYTHON)
def lint(session: Session) -> None:
    """Lint using flake8."""
    env = [
        "flake8",
        "flake8-annotations",
        "flake8-black",
        "flake8-bugbear",
        "flake8-docstrings",
        "flake8-import-order",
        "darglint",
    ]
    _setup(session)
    session.install(*env)
    session.run("flake8", ".")


@nox.session(python=PYTHON)
def mypy(session: Session) -> None:
    """Run `mypy` against `av2`.

    Args:
        session: `nox` session.
    """
    env = [
        "mypy",
        "types-pyyaml",
    ]
    _setup(session)
    session.install(*env)
    session.run("mypy", ".")


@nox.session(python=["3.8", "3.9"])  # pytype doesn't support 3.10.
def pytype(session: Session) -> None:
    """Run `pytype` against `av2`.

    Args:
        session: `nox` session.
    """
    env = [
        "git+https://github.com/google/pytype@fc08da16d4eab1deace406c52130457a7da66020",  # Fixes typing.Final issue.
        "pytest",  # Required for analysis in the test directory.
    ]
    _setup(session)
    session.install(*env)
    session.run("pytype", ".")


@nox.session(python=PYTHON)
def pytest(session: Session) -> None:
    """Run `pytest` against `av2`.

    Args:
        session: `nox` session.
    """
    env = [
        "pytest",
        "pytest-benchmark",
        "pytest-cov",
    ]
    _setup(session)
    session.install(*env)
    session.run("pytest", "tests", "--cov", "src/av2")
