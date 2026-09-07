"""Turn the `[source, p.N]` markers in a draft into a reference list or BibTeX.

The write prompt requires inline citations of the form `[rag.pdf, p.4]`, and
`papers/manifest.json` already records each corpus paper's arXiv id and title. Those two
facts together are enough to emit a real bibliography, so this module closes the loop
that `tools.format_citation()` opens.

Pure: no config, no langchain, no filesystem. The manifest is passed in as data, which is
what lets the whole thing be tested against a literal dict.
"""

import re
from dataclasses import dataclass

# `[rag.pdf, p.4]`, `[rag.pdf, p. 4]`, `[rag.pdf]`, `[https://example.com/x]`.
# The optional page group is what allows a bare source to still be counted.
_CITATION_RE = re.compile(r"\[([^\[\]]+?)(?:\s*,\s*p\.?\s*(\d+))?\]")

# Only treat a bracketed span as a citation if it names something citable. Without this,
# `[unsourced]`, `[ACADEMIC ISSUE]` and Markdown link text all become phantom references.
_CITABLE = re.compile(r"(?:\.pdf|\.md|\.txt|\.json)$|^https?://", re.IGNORECASE)

_MARKDOWN_LINK = re.compile(r"\[[^\[\]]*\]\(")


@dataclass(frozen=True)
class Citation:
    source: str
    page: int | None = None

    @property
    def is_url(self) -> bool:
        return self.source.lower().startswith(("http://", "https://"))


def extract_citations(text: str) -> list[Citation]:
    """Every citation marker in `text`, de-duplicated, in stable order.

    Ordered by source then page so a reference list is deterministic - two runs over the
    same draft must produce byte-identical output, or the export is not reproducible.
    """
    if not text:
        return []

    seen: set[Citation] = set()
    for match in _CITATION_RE.finditer(text):
        # Skip Markdown links: `[label](url)` is not a citation.
        if _MARKDOWN_LINK.match(text[match.start(): match.end() + 1]):
            continue
        source = match.group(1).strip()
        if not _CITABLE.search(source):
            continue
        page = int(match.group(2)) if match.group(2) else None
        seen.add(Citation(source, page))

    return sorted(seen, key=lambda c: (c.source.lower(), c.page if c.page is not None else -1))


def group_by_source(citations: list[Citation]) -> dict[str, list[int]]:
    """`{"rag.pdf": [1, 4, 9]}` - pages sorted, sources in first-seen order."""
    grouped: dict[str, list[int]] = {}
    for citation in citations:
        pages = grouped.setdefault(citation.source, [])
        if citation.page is not None and citation.page not in pages:
            pages.append(citation.page)
    return {source: sorted(pages) for source, pages in grouped.items()}


def _manifest_index(manifest: list[dict] | None) -> dict[str, dict]:
    return {entry["filename"]: entry for entry in (manifest or []) if "filename" in entry}


def build_reference_list(text: str, manifest: list[dict] | None = None) -> str:
    """A Markdown `## References` section for the citations in `text`.

    Falls back to the bare filename for anything not in the manifest, rather than
    dropping it - a citation the bibliography cannot resolve is still a citation, and
    silently omitting it would misrepresent the draft.
    """
    citations = extract_citations(text)
    if not citations:
        return ""

    index = _manifest_index(manifest)
    lines = ["## References", ""]

    for source, pages in group_by_source(citations).items():
        entry = index.get(source)
        if entry:
            label = entry.get("title", source)
            arxiv = entry.get("arxiv_id")
            suffix = f" (arXiv:{arxiv})" if arxiv else ""
            rendered = f"{label}{suffix}. `{source}`"
        else:
            rendered = f"`{source}`" if not source.lower().startswith("http") else source

        cited = f" — cited pp. {', '.join(str(p) for p in pages)}" if pages else ""
        lines.append(f"- {rendered}{cited}")

    return "\n".join(lines)


def _bibtex_key(entry: dict, source: str) -> str:
    if entry.get("arxiv_id"):
        return "arxiv" + entry["arxiv_id"].replace(".", "")
    return re.sub(r"[^A-Za-z0-9]+", "", source.rsplit(".", 1)[0]) or "source"


def to_bibtex(text: str, manifest: list[dict] | None = None) -> str:
    """BibTeX for every cited source the manifest can identify.

    Emits `@misc` with `eprint`/`archivePrefix` for arXiv papers, which is the form arXiv
    itself recommends and which every TeX bibliography style understands.
    """
    citations = extract_citations(text)
    if not citations:
        return ""

    index = _manifest_index(manifest)
    blocks = []

    for source in group_by_source(citations):
        entry = index.get(source)
        if entry is None:
            continue
        key = _bibtex_key(entry, source)
        fields = [f"  title = {{{entry.get('title', source)}}}"]
        if entry.get("arxiv_id"):
            fields.append(f"  eprint = {{{entry['arxiv_id']}}}")
            fields.append("  archivePrefix = {arXiv}")
        blocks.append("@misc{" + key + ",\n" + ",\n".join(fields) + "\n}")

    return "\n\n".join(blocks)
