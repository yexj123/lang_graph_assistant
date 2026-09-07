"""Turning `[source, p.N]` markers into a reference list.

The write prompt requires those markers and `papers/manifest.json` records each paper's
arXiv id and title, so a real bibliography is derivable. The hard part is not extraction
but *restraint*: `[unsourced]`, `[ACADEMIC ISSUE]` and Markdown link text all look like
citations to a naive regex, and a bibliography that invents entries is worse than none.
"""

import pytest

from bibliography import (
    Citation,
    build_reference_list,
    extract_citations,
    group_by_source,
    to_bibtex,
)

MANIFEST = [
    {"arxiv_id": "2005.11401", "filename": "rag.pdf", "title": "Retrieval-Augmented Generation"},
    {"arxiv_id": "1706.03762", "filename": "attention.pdf", "title": "Attention Is All You Need"},
]


# --- extraction ----------------------------------------------------------------

def test_extracts_source_and_page() -> None:
    assert extract_citations("As shown [rag.pdf, p.4].") == [Citation("rag.pdf", 4)]


@pytest.mark.parametrize(
    "text",
    ["[rag.pdf, p.4]", "[rag.pdf,p.4]", "[rag.pdf , p. 4]", "[rag.pdf, p 4]"],
)
def test_spacing_around_the_page_is_not_meaningful(text: str) -> None:
    assert extract_citations(text) == [Citation("rag.pdf", 4)]


def test_a_source_without_a_page_still_counts() -> None:
    assert extract_citations("see [rag.pdf]") == [Citation("rag.pdf", None)]


def test_urls_are_citations_too() -> None:
    # web_search now returns real URLs, and the research prompt asks for them.
    citations = extract_citations("per [https://example.com/paper]")

    assert citations == [Citation("https://example.com/paper", None)]
    assert citations[0].is_url


@pytest.mark.parametrize(
    "text",
    [
        "This is [unsourced].",
        "Reviewer said [ACADEMIC ISSUE] fix it.",
        "A [link](https://example.com) in prose.",
        "An array index like [0] or [i].",
        "",
    ],
)
def test_non_citations_are_ignored(text: str) -> None:
    # A phantom reference misrepresents the draft, so the bar is "looks citable or skip".
    assert extract_citations(text) == []


def test_duplicates_collapse_and_order_is_deterministic() -> None:
    text = "[rag.pdf, p.9] then [attention.pdf, p.1] then [rag.pdf, p.9] and [rag.pdf, p.2]"

    # Two runs over the same draft must produce byte-identical bibliographies.
    assert extract_citations(text) == extract_citations(text)
    assert extract_citations(text) == [
        Citation("attention.pdf", 1),
        Citation("rag.pdf", 2),
        Citation("rag.pdf", 9),
    ]


def test_grouping_collects_pages_per_source() -> None:
    grouped = group_by_source(extract_citations("[rag.pdf, p.9] [rag.pdf, p.2] [attention.pdf, p.1]"))

    assert grouped == {"attention.pdf": [1], "rag.pdf": [2, 9]}


# --- reference list -------------------------------------------------------------

def test_reference_list_resolves_titles_and_arxiv_ids() -> None:
    rendered = build_reference_list("[rag.pdf, p.1] and [rag.pdf, p.4]", MANIFEST)

    assert "## References" in rendered
    assert "Retrieval-Augmented Generation" in rendered
    assert "arXiv:2005.11401" in rendered
    assert "cited pp. 1, 4" in rendered


def test_sources_outside_the_manifest_are_listed_not_dropped() -> None:
    # Silently omitting a citation would misrepresent what the draft actually claims.
    rendered = build_reference_list("[my-own-notes.pdf, p.2]", MANIFEST)

    assert "my-own-notes.pdf" in rendered


def test_no_citations_produces_no_section() -> None:
    assert build_reference_list("A draft with no citations at all.", MANIFEST) == ""


def test_a_missing_manifest_degrades_to_bare_filenames() -> None:
    rendered = build_reference_list("[rag.pdf, p.1]", None)

    assert "rag.pdf" in rendered
    assert "arXiv" not in rendered


# --- bibtex ---------------------------------------------------------------------

def test_bibtex_uses_the_arxiv_eprint_form() -> None:
    rendered = to_bibtex("[rag.pdf, p.1]", MANIFEST)

    assert rendered.startswith("@misc{arxiv200511401,")
    assert "title = {Retrieval-Augmented Generation}" in rendered
    assert "eprint = {2005.11401}" in rendered
    assert "archivePrefix = {arXiv}" in rendered


def test_bibtex_skips_sources_it_cannot_identify() -> None:
    # A BibTeX entry with no real identifiers is not usable, so emit nothing for it.
    assert to_bibtex("[unknown.pdf, p.1]", MANIFEST) == ""


def test_bibtex_is_empty_without_citations() -> None:
    assert to_bibtex("nothing here", MANIFEST) == ""
