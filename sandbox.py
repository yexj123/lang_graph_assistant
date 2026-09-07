import os
import subprocess
import sys

# Allowlist, not blocklist: only what Windows/Python need to start a child
# interpreter are passed through. Secrets (OPENAI_API_KEY, DATABASE_URL, ...)
# are excluded by construction, so a new secret added later doesn't require
# remembering to add it to a blocklist here.
_ENV_ALLOWLIST = ("PATH", "SystemRoot", "TEMP", "TMP", "COMSPEC")
DEFAULT_TIMEOUT_SECONDS = 15
MAX_OUTPUT_CHARS = 10_000


def _scrubbed_env() -> dict[str, str]:
    return {key: os.environ[key] for key in _ENV_ALLOWLIST if key in os.environ}


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    hidden = len(text) - MAX_OUTPUT_CHARS
    return f"{text[:MAX_OUTPUT_CHARS]}\n... [truncated, {hidden} more characters]"


def run_sandboxed(code: str, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> str:
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=_scrubbed_env(),
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return f"Execution Error: timed out after {timeout}s."

    if result.returncode != 0:
        return f"Execution Error:\n{_truncate(result.stderr)}"

    output = result.stdout.strip()
    return _truncate(output) if output else "Execution successful (no output)."
