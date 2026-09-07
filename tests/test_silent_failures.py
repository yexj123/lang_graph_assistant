"""Regressions for four bugs that never raised - they just made the output quietly wrong.

Each of these had no symptom at runtime: memory stopped being enforced, `--new` gave you
the wrong thesis, a typed query vanished, and the index doubled. That is the worst
failure mode for a tool whose pitch is "it remembers", so each fix is pinned here.
"""

import operator
from typing import Annotated
from unittest import mock

import pytest
from langchain_core.documents import Document
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, END, StateGraph
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

import memory
import nodes
import session
from ingest import chunk_id
from schemas import DurableMemoryItem

_MEM_NS = ("thesis_memory", "default_user", "constraint")


def _save(store: InMemoryStore, n: int, prefix: str = "rule") -> None:
    for i in range(n):
        memory.save_durable_memory(
            store,
            DurableMemoryItem(
                category="constraint",
                title=f"{prefix}-{i}",
                content=f"content-{i}",
                status="enforced",
            ),
        )


# --- 1. durable memory was silently capped at the store's default limit=10 ------

def test_memories_beyond_the_stores_default_page_are_still_injected() -> None:
    # BaseStore.search defaults to limit=10. Constraint #11 used to vanish from every
    # prompt with no error, so a rule the user explicitly saved stopped being enforced.
    store = InMemoryStore()
    _save(store, 25)

    rendered = memory.get_durable_memories(store)

    assert "rule-0" in rendered
    assert "rule-10" in rendered, "the 11th constraint was dropped - the limit=10 bug is back"
    assert "rule-24" in rendered


def test_pagination_reports_the_true_total() -> None:
    store = InMemoryStore()
    _save(store, 250)  # more than one _PAGE_SIZE

    assert memory.count_durable_memories(store) == 250


def test_truncation_is_announced_in_the_prompt_text() -> None:
    # Capping is fine; capping silently is the bug. If the block is trimmed, the model
    # and the operator both have to be able to see that it was.
    store = InMemoryStore()
    _save(store, memory.MAX_INJECTED_MEMORIES + 7)

    rendered = memory.get_durable_memories(store)

    assert "7 further constraint(s) not shown" in rendered
    assert str(memory.MAX_INJECTED_MEMORIES + 7) in rendered


def test_no_truncation_notice_when_everything_fits() -> None:
    store = InMemoryStore()
    _save(store, 3)

    assert "not shown" not in memory.get_durable_memories(store)


def test_categories_are_derived_from_the_schema() -> None:
    # A hardcoded copy silently stops reading any category added to DecisionCategory.
    from typing import get_args
    from schemas import DecisionCategory

    assert set(memory.CATEGORIES) == set(get_args(DecisionCategory))


# --- 2. --new silently resurrected the previous thesis -------------------------

def test_archiving_the_proposal_lets_the_next_run_propose_a_new_thesis() -> None:
    store = InMemoryStore()
    store.put(
        session.PROPOSAL_NAMESPACE,
        session.PROPOSAL_KEY,
        {"thesis_topic": "Old topic", "research_question": "Old Q?", "outline": ["I"]},
    )

    archive_key = session.archive_active_proposal(store)

    assert archive_key and archive_key.startswith("archived_proposal_")
    # Nothing destroyed: the old proposal is still retrievable under its archive key.
    assert store.get(session.PROPOSAL_NAMESPACE, archive_key).value["thesis_topic"] == "Old topic"
    assert store.get(session.PROPOSAL_NAMESPACE, session.PROPOSAL_KEY) is None


def test_proposal_node_generates_fresh_after_an_archive() -> None:
    """The end-to-end point of --new-project: a new idea must not inherit the old thesis."""
    from schemas import ThesisProposal

    store = InMemoryStore()
    store.put(
        session.PROPOSAL_NAMESPACE,
        session.PROPOSAL_KEY,
        {"thesis_topic": "Old topic", "research_question": "Old Q?", "outline": ["I"]},
    )

    stub = mock.MagicMock()
    stub.with_structured_output.return_value.invoke.return_value = ThesisProposal(
        thesis_topic="Brand new topic", research_question="New Q?", outline=["A"]
    )

    # Before archiving, a fresh thread silently gets the old thesis back.
    with mock.patch.object(nodes, "get_model", lambda: stub):
        stale = nodes.proposal_node({"user_query": "something else entirely"}, store)
    assert stale["thesis_topic"] == "Old topic"

    session.archive_active_proposal(store)

    with mock.patch.object(nodes, "get_model", lambda: stub):
        fresh = nodes.proposal_node({"user_query": "something else entirely"}, store)
    assert fresh["thesis_topic"] == "Brand new topic"


def test_archiving_when_there_is_no_proposal_is_a_noop() -> None:
    assert session.archive_active_proposal(InMemoryStore()) is None


def test_nodes_and_session_agree_on_where_the_proposal_lives() -> None:
    # These were separate literals; a drift would make --new-project archive nothing.
    assert nodes.PROPOSAL_NAMESPACE == session.PROPOSAL_NAMESPACE
    assert nodes.PROPOSAL_KEY == session.PROPOSAL_KEY


# --- 3. resuming at the approval gate swallowed the first typed query -----------

class _GateState(TypedDict):
    log: Annotated[list, operator.add]
    user_query: str


def _gate_graph():
    def gate(state: _GateState) -> dict:
        reply = interrupt({"ask": "approve?"})
        return {"log": [f"resumed:{reply}"]}

    builder = StateGraph(_GateState)
    builder.add_node("gate", gate)
    builder.add_edge(START, "gate")
    builder.add_edge("gate", END)
    return builder.compile(checkpointer=InMemorySaver())


def test_invoking_with_new_input_at_a_gate_does_not_act_on_it() -> None:
    """Characterises the LangGraph behaviour that made the bug silent.

    This is why main.py must check .next before prompting: the query is written into
    state, the interrupt re-fires, and the node body never runs. Nothing raises.
    """
    graph = _gate_graph()
    config = {"configurable": {"thread_id": "t"}}
    graph.invoke({"log": [], "user_query": "first"}, config=config)

    graph.invoke({"log": [], "user_query": "please add a section"}, config=config)

    snapshot = graph.get_state(config)
    assert snapshot.next == ("gate",), "still parked at the gate"
    assert snapshot.values["log"] == [], "the node never ran, so the query did nothing"
    assert snapshot.values["user_query"] == "please add a section", "but it was stored anyway"


def test_resuming_through_the_gate_consumes_the_reply() -> None:
    graph = _gate_graph()
    config = {"configurable": {"thread_id": "t"}}
    graph.invoke({"log": [], "user_query": "first"}, config=config)

    graph.invoke(Command(resume="approve"), config=config)

    snapshot = graph.get_state(config)
    assert snapshot.next == ()
    assert snapshot.values["log"] == ["resumed:approve"]


def test_cli_services_a_pending_gate_before_prompting() -> None:
    import main

    assert hasattr(main, "_service_interrupts"), "gate handling must be callable at startup"
    source = __import__("inspect").getsource(main.run_cli)
    assert "snapshot.next" in source, "run_cli must check .next before prompting for a query"


# --- 4. ingest.py duplicated the whole corpus on every run ---------------------

def _chunk(text: str, source: str = "a.pdf", page: int = 0) -> Document:
    return Document(page_content=text, metadata={"source": source, "page": page})


def test_the_same_chunk_gets_the_same_id_across_runs() -> None:
    # Fresh UUIDs per run were why a second `python ingest.py` doubled the index.
    assert chunk_id(_chunk("identical text")) == chunk_id(_chunk("identical text"))


def test_ids_differ_by_text_page_and_source() -> None:
    base = chunk_id(_chunk("text", source="a.pdf", page=0))

    assert chunk_id(_chunk("other", source="a.pdf", page=0)) != base
    # Repeated boilerplate (running headers) is identical text on different pages;
    # hashing text alone would collapse those and silently drop real chunks.
    assert chunk_id(_chunk("text", source="a.pdf", page=1)) != base
    assert chunk_id(_chunk("text", source="b.pdf", page=0)) != base


def test_ingestion_passes_deterministic_ids_to_the_vector_store(tmp_path, monkeypatch) -> None:
    import ingest

    pdf_dir = tmp_path / "papers"
    pdf_dir.mkdir()
    (pdf_dir / "fake.pdf").write_bytes(b"%PDF-1.4 stub")
    monkeypatch.setattr(ingest, "PAPERS_DIR", pdf_dir)

    chunks = [_chunk("alpha"), _chunk("beta", page=1)]
    monkeypatch.setattr(ingest, "PyPDFLoader", lambda path: mock.MagicMock(load=lambda: chunks))
    monkeypatch.setattr(
        ingest, "RecursiveCharacterTextSplitter",
        lambda **kw: mock.MagicMock(split_documents=lambda docs: docs),
    )

    store = mock.MagicMock()
    with mock.patch.object(ingest, "get_vector_store", lambda: store):
        ingest.run_ingestion()

    _, kwargs = store.add_documents.call_args
    assert kwargs["ids"] == [chunk_id(c) for c in chunks]
    store.delete_collection.assert_not_called()


def test_rebuild_drops_the_collection_first(tmp_path, monkeypatch) -> None:
    import ingest

    pdf_dir = tmp_path / "papers"
    pdf_dir.mkdir()
    (pdf_dir / "fake.pdf").write_bytes(b"%PDF-1.4 stub")
    monkeypatch.setattr(ingest, "PAPERS_DIR", pdf_dir)
    monkeypatch.setattr(ingest, "PyPDFLoader", lambda path: mock.MagicMock(load=lambda: [_chunk("a")]))
    monkeypatch.setattr(
        ingest, "RecursiveCharacterTextSplitter",
        lambda **kw: mock.MagicMock(split_documents=lambda docs: docs),
    )

    store = mock.MagicMock()
    with mock.patch.object(ingest, "get_vector_store", lambda: store):
        ingest.run_ingestion(rebuild=True)

    store.delete_collection.assert_called_once()
    store.create_collection.assert_called_once()


# --- the CLI surface ----------------------------------------------------------

def test_new_is_still_accepted_as_an_alias_for_new_thread() -> None:
    from main import build_arg_parser

    parser = build_arg_parser()

    assert parser.parse_args(["--new"]).new_thread is True
    assert parser.parse_args(["--new"]).new_project is False
    assert parser.parse_args(["--new-thread"]).new_thread is True
    assert parser.parse_args(["--new-project"]).new_project is True
    assert parser.parse_args([]).new_thread is False
