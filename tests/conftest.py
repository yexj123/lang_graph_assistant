"""Shared safety net for the test suite.

There is deliberately very little here - tests in this project use stock pytest plus real
in-memory stores rather than a fixture layer. The one thing that has to be global is
below.
"""

import pytest

import export


@pytest.fixture(autouse=True)
def _isolate_draft_exports(tmp_path, monkeypatch):
    """Redirect every draft export into the test's own tmp_path.

    `human_approval_node` writes a file on the approve branch, so *any* test that approves
    a draft exports one. Several such tests predate the export feature and do not patch
    the destination - and they were silently filling the repository's real `drafts/`
    directory with fixture output.

    Autouse rather than opt-in because the failure mode is invisible: the tests pass
    either way, and the only symptom is litter in someone's working tree. Tests that care
    about the destination still pass `out_dir` explicitly, which takes precedence.
    """
    monkeypatch.setattr(export, "DEFAULT_OUTPUT_DIR", tmp_path / "drafts")
