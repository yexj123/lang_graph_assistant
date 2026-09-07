"""Write finished drafts to disk.

Until this existed, an approved thesis lived only in a Postgres checkpoint blob and in
terminal scrollback: `write_node` returns `{"draft": ...}` into graph state, the CLI
echoes it at the approval gate, and nothing ever touched the filesystem. For a tool whose
purpose is drafting a thesis end to end, there was no way to get the thesis out.

Kept free of config/langchain/langgraph imports, on the same reasoning as `sandbox.py`
and `hitl.py`: filename generation and file writing are fully unit-testable without a
database, and the rules below (slugging, collision handling, the front matter) are exactly
the sort of thing that is easy to get subtly wrong and cheap to pin down.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from bibliography import build_reference_list
from render import EXTENSIONS, render

DEFAULT_OUTPUT_DIR = Path("drafts")
# Mutable at runtime by `main.py --format`; read at call time, never bound as a default.
DEFAULT_FORMAT = "md"
MANIFEST_PATH = Path("papers/manifest.json")
FILENAME_TIMESTAMP = "%Y%m%dT%H%M%SZ"
MAX_SLUG_LENGTH = 60
_FALLBACK_SLUG = "thesis"

_NON_SLUG = re.compile(r"[^a-z0-9]+")


def slugify(text: str, max_length: int = MAX_SLUG_LENGTH) -> str:
    """A filesystem-safe stem for a thesis topic.

    Thesis topics are long, punctuated and often contain characters Windows rejects in
    filenames (`:` above all), so this keeps ASCII alphanumerics only and truncates on a
    word boundary. Falls back to a fixed stem rather than producing an empty filename.
    """
    slug = _NON_SLUG.sub("-", text.strip().lower()).strip("-")
    if not slug:
        return _FALLBACK_SLUG
    if len(slug) <= max_length:
        return slug
    # Truncate at the last separator inside the budget so the name stays readable.
    truncated = slug[:max_length]
    head, sep, _ = truncated.rpartition("-")
    return (head if sep and head else truncated).strip("-") or _FALLBACK_SLUG


def draft_filename(thesis_topic: str, when: datetime | None = None) -> str:
    """`<topic-slug>-<utc-timestamp>.md`.

    The timestamp is what makes repeated exports non-destructive: approving, revising and
    approving again leaves both versions on disk instead of overwriting the first.
    """
    stamp = (when or datetime.now(timezone.utc)).strftime(FILENAME_TIMESTAMP)
    return f"{slugify(thesis_topic)}-{stamp}.md"


def build_front_matter(thesis_topic: str, research_question: str, when: datetime) -> str:
    """A short human-readable header, so an exported file is self-describing."""
    lines = [f"# {thesis_topic.strip() or 'Untitled thesis'}", ""]
    if research_question.strip():
        lines += [f"**Research question:** {research_question.strip()}", ""]
    lines += [f"*Exported {when.strftime('%Y-%m-%d %H:%M:%S')} UTC*", "", "---", ""]
    return "\n".join(lines)


def load_corpus_manifest() -> list[dict]:
    """The corpus manifest, for resolving citations to titles. [] if unavailable.

    Best-effort by design: a missing or malformed manifest degrades the bibliography to
    bare filenames, which is still correct - it must never block an export.
    """
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["papers"]
    except (OSError, ValueError, KeyError):
        return []


def write_draft(
    draft: str,
    thesis_topic: str = "",
    research_question: str = "",
    out_dir: Path | str | None = None,
    when: datetime | None = None,
    fmt: str = "",
    manifest: list[dict] | None = None,
) -> Path:
    """Write `draft` to `<out_dir>/<slug>-<timestamp>.md` and return the path.

    `out_dir` defaults to DEFAULT_OUTPUT_DIR, resolved *at call time* rather than as a
    default argument. Python binds default arguments once at definition, so
    `out_dir=DEFAULT_OUTPUT_DIR` would make the module constant impossible to override -
    including from a test, which would then quietly write into the real drafts/ directory.

    Raises ValueError on an empty draft: writing a zero-byte file that looks like a
    successful export is worse than failing.
    """
    if not draft or not draft.strip():
        raise ValueError("refusing to export an empty draft")

    when = when or datetime.now(timezone.utc)
    fmt = fmt or DEFAULT_FORMAT
    directory = Path(out_dir) if out_dir else DEFAULT_OUTPUT_DIR
    if manifest is None:
        manifest = load_corpus_manifest()
    directory.mkdir(parents=True, exist_ok=True)

    stem = Path(draft_filename(thesis_topic, when)).stem
    path = directory / f"{stem}{EXTENSIONS.get(fmt, '.md')}"
    # Same-second exports would otherwise clobber each other.
    counter = 2
    while path.exists():
        path = directory / f"{stem}-{counter}{path.suffix}"
        counter += 1

    body = build_front_matter(thesis_topic, research_question, when) + draft.rstrip() + "\n"

    # Append a References section derived from the draft's own [source, p.N] markers,
    # unless the writer already produced one. Nothing is invented: every entry traces to
    # a citation that is actually in the text.
    if "## References" not in body:
        references = build_reference_list(body, manifest)
        if references:
            body += "\n" + references + "\n"

    return render(body, fmt, path, title=thesis_topic)
