"""Rendering a Markdown draft into submission formats.

md and tex are pure transforms and fully tested here. docx and pdf depend on tooling that
may not be installed, so those tests assert the *contract on absence*: a clear, actionable
RenderError rather than an ImportError or a silent empty file.
"""

from pathlib import Path

import pytest

import render
from render import (
    RenderError,
    escape_tex,
    markdown_to_docx,
    markdown_to_latex,
    markdown_to_pdf,
)


# --- LaTeX escaping -------------------------------------------------------------

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("50% faster", r"50\% faster"),
        ("a & b", r"a \& b"),
        ("cost $5", r"cost \$5"),
        ("snake_case", r"snake\_case"),
        ("#hashtag", r"\#hashtag"),
        ("{braces}", r"\{braces\}"),
    ],
)
def test_special_characters_are_escaped(raw: str, expected: str) -> None:
    assert escape_tex(raw) == expected


def test_backslash_is_escaped_without_corrupting_the_others() -> None:
    # Order matters: substituting the backslash last would re-escape the replacements
    # generated for %, & and friends.
    assert escape_tex(r"a\b & c") == r"a\textbackslash{}b \& c"


# --- Markdown to LaTeX ----------------------------------------------------------

def test_headings_map_to_sectioning_commands() -> None:
    tex = markdown_to_latex("# One\n## Two\n### Three\n#### Four")

    assert r"\section{One}" in tex
    assert r"\subsection{Two}" in tex
    assert r"\subsubsection{Three}" in tex
    assert r"\paragraph{Four}" in tex


def test_bullets_and_numbers_become_the_right_environments() -> None:
    tex = markdown_to_latex("- a\n- b\n\n1. x\n2. y")

    assert r"\begin{itemize}" in tex and r"\end{itemize}" in tex
    assert r"\begin{enumerate}" in tex and r"\end{enumerate}" in tex
    assert tex.count(r"\item") == 4


def test_a_list_is_closed_before_the_next_heading() -> None:
    # An unclosed environment makes the document fail to compile at all.
    tex = markdown_to_latex("- a\n\n## Next")

    assert tex.index(r"\end{itemize}") < tex.index(r"\subsection{Next}")


def test_inline_emphasis_survives_escaping() -> None:
    tex = markdown_to_latex("**bold** and *italic* and `code`")

    assert r"\textbf{bold}" in tex
    assert r"\emph{italic}" in tex
    assert r"\texttt{code}" in tex


def test_horizontal_rules_do_not_become_stray_em_dashes() -> None:
    # `---` passed through literally renders as an em-dash floating on its own line.
    tex = markdown_to_latex("above\n\n---\n\nbelow")

    assert "hrulefill" in tex


def test_the_document_is_structurally_complete() -> None:
    tex = markdown_to_latex("Body text.", title="My Thesis", author="Me")

    assert tex.startswith(r"\documentclass")
    assert r"\title{My Thesis}" in tex
    assert r"\author{Me}" in tex
    assert tex.index(r"\begin{document}") < tex.index("Body text.") < tex.index(r"\end{document}")


def test_a_title_with_special_characters_is_escaped() -> None:
    assert r"\title{RAG \& Accuracy}" in markdown_to_latex("x", title="RAG & Accuracy")


# --- dispatch --------------------------------------------------------------------

def test_render_writes_markdown_unchanged(tmp_path: Path) -> None:
    target = tmp_path / "out.md"

    render.render("# Hi\n\nBody.", "md", target)

    assert target.read_text(encoding="utf-8") == "# Hi\n\nBody."


def test_render_creates_missing_directories(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "out.tex"

    render.render("# Hi", "tex", target)

    assert target.exists()


def test_an_unknown_format_names_the_valid_ones() -> None:
    with pytest.raises(RenderError) as excinfo:
        render.render("x", "epub", Path("out.epub"))

    for fmt in render.FORMATS:
        assert fmt in str(excinfo.value)


def test_every_declared_format_has_an_extension() -> None:
    assert set(render.FORMATS) == set(render.EXTENSIONS)


# --- optional toolchains ----------------------------------------------------------

def test_docx_without_python_docx_says_how_to_install_it(tmp_path: Path, monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "docx":
            raise ImportError("no docx")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)

    with pytest.raises(RenderError, match="pip install python-docx"):
        markdown_to_docx("# x", tmp_path / "out.docx")


def test_pdf_without_a_toolchain_explains_the_alternatives(tmp_path: Path, monkeypatch) -> None:
    # Rather than bundle a Python PDF renderer that would do a poor job on a thesis, the
    # contract is: use pandoc/pdflatex if present, otherwise say so and point at `tex`.
    monkeypatch.setattr(render, "pdf_toolchain", lambda: None)

    with pytest.raises(RenderError) as excinfo:
        markdown_to_pdf("# x", tmp_path / "out.pdf")

    message = str(excinfo.value)
    assert "pandoc" in message and "pdflatex" in message and "tex" in message


def test_pdf_toolchain_detection_reports_what_is_installed() -> None:
    # Whatever this machine has, the answer must be one of the two or None.
    assert render.pdf_toolchain() in (None, "pandoc", "pdflatex")
