"""Structural checks on the RAG golden set and the corpus manifest.

The previous golden set was two hand-written Q&A pairs whose reference answers cited
nothing, about papers that were not in the repo and were named nowhere - so there was
no way to tell whether a low score meant a bad pipeline or a bad ground truth.

These tests make the ground truth auditable. They cost nothing to run: no API calls,
no database. The quote check needs the PDFs, so it skips when the corpus has not been
fetched rather than failing on a clean clone.
"""

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = Path(__file__).parent / "golden_set.json"
MANIFEST_PATH = REPO_ROOT / "papers" / "manifest.json"

GOLDENS = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))["goldens"]
MANIFEST = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["papers"]
CORPUS_FILENAMES = {entry["filename"] for entry in MANIFEST}

_IDS = [g["source"] + f"-p{g['page']}" for g in GOLDENS]


def test_the_golden_set_is_large_enough_to_be_a_measurement() -> None:
    # Two samples against a stochastic LLM judge is noise, not a measurement.
    assert len(GOLDENS) >= 15, f"only {len(GOLDENS)} goldens"


def test_questions_are_unique() -> None:
    questions = [g["question"] for g in GOLDENS]
    assert len(set(questions)) == len(questions), "duplicate question in the golden set"


def test_every_manifest_entry_is_exercised_by_at_least_one_golden() -> None:
    # A paper nobody asks about is dead weight in the index and skews retrieval.
    used = {g["source"] for g in GOLDENS}
    assert used == CORPUS_FILENAMES, f"unused: {CORPUS_FILENAMES - used}, unknown: {used - CORPUS_FILENAMES}"


@pytest.mark.parametrize("golden", GOLDENS, ids=_IDS)
def test_each_golden_is_attributed_to_a_real_corpus_paper(golden: dict) -> None:
    assert golden["source"] in CORPUS_FILENAMES, f"{golden['source']} is not in the manifest"
    # 1-indexed, matching tools.format_citation output; page 0 would be a citation bug.
    assert isinstance(golden["page"], int) and golden["page"] >= 1
    assert golden["reference"].strip(), "empty reference answer"
    assert golden["source_quote"].strip(), "reference answer with no supporting quote"


@pytest.mark.parametrize("entry", MANIFEST, ids=[e["arxiv_id"] for e in MANIFEST])
def test_manifest_entries_are_complete_and_checksummed(entry: dict) -> None:
    assert re.fullmatch(r"\d{4}\.\d{4,5}", entry["arxiv_id"]), entry["arxiv_id"]
    assert entry["filename"].endswith(".pdf")
    assert entry["title"].strip()
    # Without a checksum "the same corpus" is claimed, not verifiable.
    assert re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256") or ""), "missing or malformed sha256"


@pytest.mark.parametrize("golden", GOLDENS, ids=_IDS)
def test_the_supporting_quote_really_appears_on_that_page(golden: dict) -> None:
    """The check that makes the ground truth trustworthy rather than plausible."""
    pdf_path = REPO_ROOT / "papers" / golden["source"]
    if not pdf_path.exists():
        pytest.skip(f"{golden['source']} not fetched - run `python fetch_papers.py`")

    from pypdf import PdfReader

    page_text = PdfReader(str(pdf_path)).pages[golden["page"] - 1].extract_text() or ""
    # PDF extraction inserts stray whitespace mid-word, so compare on collapsed spacing.
    haystack = re.sub(r"\s+", " ", page_text)
    needle = re.sub(r"\s+", " ", golden["source_quote"])

    assert needle in haystack, (
        f"quote not found on p.{golden['page']} of {golden['source']}:\n  {needle!r}"
    )
