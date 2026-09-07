"""Startup guarantees: nothing touches the network until it is actually needed.

config.py used to construct every client at module scope. `ChatOpenAI` raises
without OPENAI_API_KEY and `PGVector` opens a Postgres connection in its
constructor, so importing any module in this project required both to be present.
`python main.py --help` inherited that and crashed after ~25 seconds.

The subprocess tests below run with a scrubbed environment (no OPENAI_API_KEY, and
a DATABASE_URL pointing at a closed port) so they fail loudly if that regresses,
rather than passing by accident on a developer machine that has both configured.
"""

import os
import subprocess
import sys
import time

import pytest

# Deliberately unroutable: nothing listens here, so any real connection attempt
# stalls until connect_timeout and blows the elapsed-time assertions.
_DEAD_DB = "postgresql://postgres:postgres@127.0.0.1:1/definitely_not_here"

# The primary guard is returncode: against a closed port a real connection attempt is
# refused outright, and a missing key raises in ChatOpenAI, so either regression exits
# non-zero. The clock is a backstop for a DATABASE_URL that hangs instead of refusing,
# so the budget only has to sit clearly above a cold langchain import (~14s).
_IMPORT_BUDGET_SECONDS = 45


def _run_isolated(code_or_args: list[str]) -> tuple[subprocess.CompletedProcess, float]:
    """Run python in a child process with no OpenAI key and a dead database."""
    env = {k: v for k, v in os.environ.items() if k not in ("OPENAI_API_KEY", "DATABASE_URL")}
    env["DATABASE_URL"] = _DEAD_DB

    start = time.time()
    result = subprocess.run(
        [sys.executable, *code_or_args],
        capture_output=True,
        text=True,
        env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        timeout=90,
        stdin=subprocess.DEVNULL,
    )
    return result, time.time() - start


# graph transitively imports config, nodes and tools, so it covers that whole chain;
# ingest reaches config by its own path, and main must stay import-light for --help.
@pytest.mark.parametrize("module", ["graph", "ingest", "main"])
def test_importing_a_module_needs_neither_a_database_nor_an_api_key(module: str) -> None:
    result, elapsed = _run_isolated(["-c", f"import {module}"])

    assert result.returncode == 0, f"import {module} failed:\n{result.stderr}"
    assert elapsed < _IMPORT_BUDGET_SECONDS, f"import {module} took {elapsed:.1f}s - did it connect?"


def test_help_works_without_any_configuration() -> None:
    # The original bug: main.py imported config and graph at module scope, so --help
    # connected to Postgres and died on a SQLAlchemy traceback instead of printing help.
    result, elapsed = _run_isolated(["main.py", "--help"])

    assert result.returncode == 0, f"--help failed:\n{result.stderr}"
    assert "--new" in result.stdout
    assert elapsed < _IMPORT_BUDGET_SECONDS, f"--help took {elapsed:.1f}s - did it connect?"


# --- accessor behaviour ------------------------------------------------------

@pytest.mark.parametrize(
    "accessor",
    ["get_model", "get_embeddings", "get_pool", "get_vector_store", "get_checkpointer", "get_store"],
)
def test_every_client_is_exposed_as_a_lazy_accessor(accessor: str) -> None:
    import config

    # A plain module attribute would mean eager construction; a callable means deferred.
    assert callable(getattr(config, accessor))


def test_accessors_are_cached_so_callers_share_one_instance() -> None:
    import config

    # Every module reaches these through config rather than building its own client,
    # so the cache is what makes "one pool, one model" true.
    for accessor in ("get_model", "get_embeddings", "get_pool", "get_vector_store"):
        assert hasattr(getattr(config, accessor), "cache_info"), f"{accessor} is not lru_cached"


def test_startup_error_message_names_the_unreachable_database() -> None:
    import config
    from config import startup_error_message

    message = startup_error_message(OSError("connection timeout expired"))

    assert config.DB_URI in message
    assert "Postgres" in message
    # The original text must survive: the hint narrows the diagnosis, it must not hide it.
    assert "connection timeout expired" in message


def test_startup_error_message_explains_a_missing_api_key() -> None:
    from config import startup_error_message

    message = startup_error_message(RuntimeError("Missing credentials. Please pass an `api_key`"))

    assert "OPENAI_API_KEY" in message
    assert ".env" in message
