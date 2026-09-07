"""Getting the thesis out of the machine.

Before export.py, `write_node` returned the draft into graph state and nothing ever wrote
a file: an approved thesis survived only inside a Postgres checkpoint blob and in terminal
scrollback. These tests cover the filename rules (which are easy to get subtly wrong on
Windows), the non-destructive write, and the two paths through human_approval_node that
trigger an export.
"""

from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import pytest
from langgraph.graph import END
from langgraph.store.memory import InMemoryStore

import export
import nodes
from export import build_front_matter, draft_filename, slugify, write_draft

_WHEN = datetime(2026, 9, 7, 15, 3, 4, tzinfo=timezone.utc)


def _model(structured_result=None, invoke_result=None) -> mock.MagicMock:
    m = mock.MagicMock(name="model")
    m.with_structured_output.return_value.invoke.return_value = structured_result
    m.invoke.return_value = invoke_result
    return m


# --- slugify -------------------------------------------------------------------

@pytest.mark.parametrize(
    ("topic", "expected"),
    [
        ("Retrieval-Augmented Generation", "retrieval-augmented-generation"),
        ("  Spaces  Everywhere  ", "spaces-everywhere"),
        # Windows rejects ':' outright in filenames, and thesis titles are full of them.
        ("Attention: A Study", "attention-a-study"),
        ("C++ & Rust: 50% faster?", "c-rust-50-faster"),
        ("---leading and trailing---", "leading-and-trailing"),
        # Non-ASCII is dropped rather than transliterated; the timestamp keeps it unique.
        ("Überlegungen", "berlegungen"),
        # Never return an empty stem - that would produce a file called just "-<stamp>.md".
        ("", "thesis"),
        ("!!!", "thesis"),
        ("   ", "thesis"),
    ],
)
def test_slugify(topic: str, expected: str) -> None:
    assert slugify(topic) == expected


def test_slugify_truncates_on_a_word_boundary() -> None:
    slug = slugify("the effect of retrieval augmentation on factual accuracy in large language models")

    assert len(slug) <= export.MAX_SLUG_LENGTH
    assert not slug.endswith("-")
    # Truncated at a separator, so the tail is a whole word rather than a fragment.
    assert slug.split("-")[-1] in "the effect of retrieval augmentation on factual accuracy in large".split()


def test_slugify_handles_a_single_overlong_word() -> None:
    slug = slugify("a" * 200)

    assert slug == "a" * export.MAX_SLUG_LENGTH


# --- filenames -----------------------------------------------------------------

def test_draft_filename_is_slug_plus_utc_timestamp() -> None:
    assert draft_filename("Attention: A Study", _WHEN) == "attention-a-study-20260907T150304Z.md"


def test_front_matter_names_the_topic_and_question() -> None:
    header = build_front_matter("My Topic", "Does X cause Y?", _WHEN)

    assert header.startswith("# My Topic")
    assert "**Research question:** Does X cause Y?" in header
    assert "2026-09-07" in header


def test_front_matter_omits_an_absent_research_question() -> None:
    assert "Research question" not in build_front_matter("T", "   ", _WHEN)


def test_front_matter_falls_back_to_a_title() -> None:
    assert build_front_matter("", "", _WHEN).startswith("# Untitled thesis")


# --- write_draft ----------------------------------------------------------------

def test_write_draft_creates_the_directory_and_returns_the_path(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "drafts"

    path = write_draft("# Chapter 1\nBody.", "My Thesis", "Q?", out_dir=out, when=_WHEN)

    assert path == out / "my-thesis-20260907T150304Z.md"
    assert path.exists()
    body = path.read_text(encoding="utf-8")
    assert "# My Thesis" in body
    assert "**Research question:** Q?" in body
    assert "# Chapter 1\nBody." in body
    assert body.endswith("\n")


def test_repeated_exports_do_not_overwrite_each_other(tmp_path: Path) -> None:
    # Approving, revising and approving again must leave both versions on disk. Same
    # timestamp is the worst case, so force it.
    first = write_draft("v1", "T", out_dir=tmp_path, when=_WHEN)
    second = write_draft("v2", "T", out_dir=tmp_path, when=_WHEN)

    assert first != second
    assert first.read_text(encoding="utf-8").endswith("v1\n")
    assert second.read_text(encoding="utf-8").endswith("v2\n")
    assert len(list(tmp_path.glob("*.md"))) == 2


@pytest.mark.parametrize("draft", ["", "   ", "\n\t "])
def test_an_empty_draft_is_refused(tmp_path: Path, draft: str) -> None:
    # A zero-byte file that looks like a successful export is worse than a loud failure.
    with pytest.raises(ValueError, match="empty draft"):
        write_draft(draft, "T", out_dir=tmp_path)

    assert list(tmp_path.glob("*.md")) == []


def test_the_file_is_written_with_lf_and_utf8(tmp_path: Path) -> None:
    path = write_draft("café naïve — dash", "T", out_dir=tmp_path, when=_WHEN)

    raw = path.read_bytes()
    assert "café".encode("utf-8") in raw
    assert b"\r\n" not in raw


# --- the node paths that export --------------------------------------------------

def test_approving_writes_the_draft_to_disk(tmp_path: Path) -> None:
    store = InMemoryStore()
    state = {
        "draft": "# Final\nBody.",
        "thesis_topic": "My Thesis",
        "research_question": "Q?",
        "research_notes": ["evidence"],
    }

    with mock.patch.object(export, "DEFAULT_OUTPUT_DIR", tmp_path), \
         mock.patch.object(nodes, "interrupt", return_value="approve"):
        command = nodes.human_approval_node(state, store)

    assert command.goto == END
    written = list(tmp_path.glob("*.md"))
    assert len(written) == 1
    assert command.update["draft_path"] == str(written[0])
    assert "# Final" in written[0].read_text(encoding="utf-8")


def test_export_command_saves_a_snapshot_and_asks_again(tmp_path: Path) -> None:
    state = {"draft": "work in progress", "thesis_topic": "T"}

    with mock.patch.object(export, "DEFAULT_OUTPUT_DIR", tmp_path), \
         mock.patch.object(nodes, "interrupt", return_value="export"):
        command = nodes.human_approval_node(state, InMemoryStore())

    # Exporting is not a decision: it must return to the gate, not advance the graph.
    assert command.goto == "human_approval_node"
    assert len(list(tmp_path.glob("*.md"))) == 1


def test_export_accepts_an_explicit_destination(tmp_path: Path) -> None:
    target = tmp_path / "somewhere-else"

    with mock.patch.object(nodes, "interrupt", return_value=f"export: {target}"):
        command = nodes.human_approval_node({"draft": "d", "thesis_topic": "T"}, InMemoryStore())

    assert command.goto == "human_approval_node"
    assert len(list(target.glob("*.md"))) == 1


def test_a_failed_export_reports_but_never_loses_the_approval() -> None:
    # Disk full, bad permissions, empty draft - none of it may crash the gate or discard
    # an approval the operator already gave.
    store = InMemoryStore()

    with mock.patch.object(nodes, "write_draft", side_effect=OSError("disk full")), \
         mock.patch.object(nodes, "interrupt", return_value="approve"):
        command = nodes.human_approval_node({"draft": "d", "research_question": "Q?"}, store)

    assert command.goto == END
    assert command.update["review_status"] == "final_approved"
    assert any("[EXPORT FAILED]" in f for f in command.update["review_feedback"])


def test_exporting_an_empty_draft_is_reported_not_raised() -> None:
    with mock.patch.object(nodes, "interrupt", return_value="export"):
        command = nodes.human_approval_node({"draft": "", "thesis_topic": "T"}, InMemoryStore())

    assert command.goto == "human_approval_node"
    assert any("[EXPORT FAILED]" in f for f in command.update["review_feedback"])
