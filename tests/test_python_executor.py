import subprocess
import time
from unittest import mock

from sandbox import run_sandboxed


def test_returns_stdout_for_successful_code() -> None:
    result = run_sandboxed("print('hello from sandbox')")
    assert "hello from sandbox" in result


def test_returns_success_message_when_no_output() -> None:
    result = run_sandboxed("x = 1 + 1")
    assert result == "Execution successful (no output)."


def test_returns_error_for_raising_code() -> None:
    result = run_sandboxed("raise ValueError('boom')")
    assert "Execution Error" in result
    assert "boom" in result


def test_scrubs_environment_variables() -> None:
    # Deliberately checks presence, never prints the real value: this parent process
    # has a real OPENAI_API_KEY set, and a failing assertion's diff must not leak it.
    result = run_sandboxed("import os; print('OPENAI_API_KEY' in os.environ)")
    assert result.strip() == "False"


def test_kills_infinite_loop_after_timeout() -> None:
    start = time.time()
    result = run_sandboxed("while True: pass", timeout=2)
    elapsed = time.time() - start

    assert elapsed < 10, f"took {elapsed}s, timeout was not enforced"
    assert "timed out" in result.lower()


def test_truncates_large_output() -> None:
    result = run_sandboxed("print('x' * 20000)")
    assert len(result) < 15000
    assert "truncated" in result.lower()


def test_disables_stdin_inheritance() -> None:
    # A wall-clock "does input() hang" test can't reproduce the real risk (an
    # interactive parent terminal leaking into the child) because this test
    # harness's own stdin is already non-interactive, so it would pass
    # regardless of whether the safeguard exists. Assert the safeguard itself.
    with mock.patch("sandbox.subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="ok", stderr=""
        )
        run_sandboxed("print('ok')")

    _, kwargs = mock_run.call_args
    assert kwargs.get("stdin") == subprocess.DEVNULL
