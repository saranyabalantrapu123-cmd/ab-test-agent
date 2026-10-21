"""Test setup: offline mode and an isolated, freshly generated dataset.

Environment variables are set before ``abagent`` is imported so its paths point at a
temporary directory and no test ever touches data/ or reports/ in the repo.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_TMP = Path(tempfile.mkdtemp(prefix="abagent-tests-"))
os.environ["ABAGENT_DATA_DIR"] = str(_TMP / "data")
os.environ["ABAGENT_REPORTS_DIR"] = str(_TMP / "reports")
os.environ["LLM_MODEL"] = "mock"
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

from generate_experiments import generate  # noqa: E402

from abagent.config import DATA_DIR  # noqa: E402


@pytest.fixture(scope="session")
def dataset():
    """(dataframe, ground truth) generated once per test session with the default seed."""
    df, truth = generate(0, DATA_DIR)
    return df, truth
