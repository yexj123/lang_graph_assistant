"""The evidence path: retrieval -> notes -> draft -> review.

ingest.py has always attached page metadata "for agent citation", but
search_thesis_literature dropped it and no prompt ever asked for a citation, so an
academic writing tool could not produce a traceable reference. Separately, the
reviewer was asked to judge factual gaps with no source material in its prompt.

These tests pin the contract at each hop.
"""

from unittest import mock

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langgraph.store.memory import InMemoryStore

import nodes
import tools
from schemas import ReviewDecision


def _model(structured_result=None, invoke_result=None) -> mock.MagicMock:
    m = mock.MagicMock(name="model")
    m.with_structured_output.return_value.invoke.return_value = structured_result
    m.invoke.return_value = invoke_result
    return m


# --- format_citation ---------------------------------------------------------

@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        # PyPDFLoader numbers pages from 0; a citation that says p.0 is wrong.
        ({"source": "attention.pdf", "page": 0}, "attention.pdf, p.1"),
        ({"source": "attention.pdf", "page": 11}, "attention.pdf, p.12"),
        # Non-PDF chunks carry no page - degrade to the bare source, never "p.None".
        ({"source": "notes.md"}, "notes.md"),
        ({"source": "weird.pdf", "page": "iv"}, "weird.pdf"),
        ({}, "Unknown"),
    ],
)
def test_format_citation(metadata: dict, expected: str) -> None:
    assert tools.format_citation(metadata) == expected


def test_format_citation_rejects_bool_pages() -> None:
    # bool is a subclass of int in Python, so a naive isinstance check would render
    # `page: True` as "p.2".
    assert tools.format_citation({"source": "x.pdf", "page": True}) == "x.pdf"


# --- search_thesis_literature ------------------------------------------------

def test_search_results_carry_a_page_level_citation() -> None:
    docs = [
        Document(page_content="Self-attention scales.", metadata={"source": "attention.pdf", "page": 3}),
        Document(page_content="Suffix trees compress.", metadata={"source": "cst.pdf", "page": 0}),
    ]
    store = mock.MagicMock()
    store.similarity_search.return_value = docs

    with mock.patch.object(tools, "get_vector_store", lambda: store):
        result = tools.search_thesis_literature.invoke({"query": "attention"})

    assert "attention.pdf, p.4" in result
    assert "cst.pdf, p.1" in result
    assert "Self-attention scales." in result


def test_search_reports_an_empty_index_rather_than_failing() -> None:
    store = mock.MagicMock()
    store.similarity_search.return_value = []

    with mock.patch.object(tools, "get_vector_store", lambda: store):
        result = tools.search_thesis_literature.invoke({"query": "anything"})

    assert "No matching sections" in result


# --- the prompts that carry citations through ---------------------------------

def test_research_prompt_tells_the_agent_to_keep_source_markers() -> None:
    response = AIMessage(content="finding")

    captured = mock.MagicMock(**{"invoke.return_value": response})

    with mock.patch.object(nodes, "get_model_with_tools", lambda: captured):
        nodes.research_node({"messages": []}, InMemoryStore())

    prompt = captured.invoke.call_args[0][0][0].content
    assert "[filename, p.N]" in prompt
    assert "verbatim" in prompt


def test_write_prompt_demands_inline_citations_and_forbids_inventing_them() -> None:
    stub = _model(invoke_result=AIMessage(content="draft"))

    with mock.patch.object(nodes, "get_model", lambda: stub):
        nodes.write_node({"research_question": "Q?", "research_notes": ["fact [a.pdf, p.2]"]}, InMemoryStore())

    prompt = stub.invoke.call_args[0][0][0].content
    assert "[source, p.N]" in prompt
    assert "Never invent a citation" in prompt
    assert "References" in prompt


# --- the reviewer can finally see the evidence --------------------------------

def test_reviewer_prompt_includes_the_research_notes() -> None:
    # Without this the reviewer judges prose against an outline, so every
    # 'needs_more_research' verdict it returns is unfounded.
    stub = _model(structured_result=ReviewDecision(next_step="approved", issues=[], suggestions=[]))
    state = {"draft": "d", "revision_count": 0, "research_notes": ["EVIDENCE-SENTINEL [a.pdf, p.7]"]}

    with mock.patch.object(nodes, "get_model", lambda: stub):
        nodes.reviewer_node(state)

    prompt = stub.with_structured_output.return_value.invoke.call_args[0][0]
    assert "EVIDENCE-SENTINEL [a.pdf, p.7]" in prompt
    assert "not against your own knowledge" in prompt


def test_reviewer_is_told_when_no_evidence_was_gathered() -> None:
    # A draft written with no research at all must not read as "silently fine".
    stub = _model(structured_result=ReviewDecision(next_step="approved", issues=[], suggestions=[]))

    with mock.patch.object(nodes, "get_model", lambda: stub):
        nodes.reviewer_node({"draft": "d", "revision_count": 0})

    prompt = stub.with_structured_output.return_value.invoke.call_args[0][0]
    assert "(none gathered yet)" in prompt
