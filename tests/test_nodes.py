"""Unit tests for the routing and node logic in nodes.py.

nodes.py is the part of this system that actually makes decisions. It became
straightforwardly testable once config.py's clients moved behind lazy accessors:
importing nodes.py no longer constructs an OpenAI client or opens a Postgres
connection, so these tests simply swap the accessors out.
"""

from unittest import mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END
from langgraph.store.memory import InMemoryStore

import nodes
from schemas import DurableMemoryItem, ReviewDecision, SupervisorDecision, ThesisProposal

_PROPOSAL_NS = ("thesis_workspace", "active_project")
_PROPOSAL_KEY = "current_proposal"


def _model(structured_result=None, invoke_result=None) -> mock.MagicMock:
    """Stand-in for config.model.

    `structured_result` is what `model.with_structured_output(Schema).invoke(...)`
    returns; `invoke_result` is what a plain `model.invoke(...)` returns.
    """
    m = mock.MagicMock(name="model")
    m.with_structured_output.return_value.invoke.return_value = structured_result
    m.invoke.return_value = invoke_result
    return m


def _tool_call_message(name: str = "web_search") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": {"query": "x"}, "id": "call_1"}],
    )


# --- supervisor_router -------------------------------------------------------

@pytest.mark.parametrize("destination", ["research_node", "write_node"])
def test_supervisor_router_returns_the_models_choice(destination: str) -> None:
    stub = _model(structured_result=SupervisorDecision(next_step=destination))

    with mock.patch.object(nodes, "get_model", lambda: stub):
        assert nodes.supervisor_router({"user_query": "anything"}) == destination

    stub.with_structured_output.assert_called_once_with(SupervisorDecision)


# --- research_router ---------------------------------------------------------

def test_research_router_goes_to_tools_when_the_model_asked_for_one() -> None:
    state = {"messages": [HumanMessage(content="hi"), _tool_call_message()]}

    assert nodes.research_router(state) == "tool_node"


def test_research_router_goes_to_writing_when_no_tool_was_requested() -> None:
    state = {"messages": [AIMessage(content="I have gathered enough.")]}

    assert nodes.research_router(state) == "write_node"


def test_research_router_goes_to_writing_on_empty_history() -> None:
    assert nodes.research_router({"messages": []}) == "write_node"


# --- proposal_node -----------------------------------------------------------

def test_proposal_node_is_a_noop_once_the_thesis_is_initialized() -> None:
    stub = _model()

    with mock.patch.object(nodes, "get_model", lambda: stub):
        result = nodes.proposal_node({"thesis_initialized": True}, InMemoryStore())

    assert result == {}
    stub.with_structured_output.assert_not_called()


def test_proposal_node_reloads_a_saved_proposal_without_calling_the_model() -> None:
    # NOTE: this reload also fires on a brand-new thread, which is why `main.py --new`
    # silently resurrects the previous thesis instead of starting one (REVIEW_OPUS.md
    # issue #5). The reload itself is correct for resume; the flag semantics are not.
    store = InMemoryStore()
    store.put(
        _PROPOSAL_NS,
        _PROPOSAL_KEY,
        {"thesis_topic": "Saved topic", "research_question": "Saved Q?", "outline": ["I", "II"]},
    )
    stub = _model()

    with mock.patch.object(nodes, "get_model", lambda: stub):
        result = nodes.proposal_node({"user_query": "a completely new idea"}, store)

    assert result["thesis_initialized"] is True
    assert result["thesis_topic"] == "Saved topic"
    assert result["research_question"] == "Saved Q?"
    assert result["outline"] == ["I", "II"]
    stub.with_structured_output.assert_not_called()


def test_proposal_node_generates_and_persists_a_new_proposal() -> None:
    store = InMemoryStore()
    proposal = ThesisProposal(
        thesis_topic="Generated topic",
        research_question="Generated Q?",
        outline=["Intro", "Method"],
    )
    stub = _model(structured_result=proposal)

    with mock.patch.object(nodes, "get_model", lambda: stub):
        result = nodes.proposal_node({"user_query": "my thesis idea"}, store)

    assert result["thesis_topic"] == "Generated topic"
    assert result["outline"] == ["Intro", "Method"]
    assert result["review_feedback"] == []
    # It must survive for the next session, otherwise the proposal step never gets skipped.
    assert store.get(_PROPOSAL_NS, _PROPOSAL_KEY).value["thesis_topic"] == "Generated topic"


# --- research_node -----------------------------------------------------------

def test_research_node_does_not_record_notes_while_still_calling_tools() -> None:
    response = _tool_call_message()

    with mock.patch.object(nodes, "get_model_with_tools", lambda: mock.MagicMock(**{"invoke.return_value": response})):
        result = nodes.research_node({"messages": []}, InMemoryStore())

    assert result["messages"] == [response]
    assert "research_notes" not in result


def test_research_node_records_notes_from_a_final_answer() -> None:
    response = AIMessage(content="Finding: transformers scale with data.")

    with mock.patch.object(nodes, "get_model_with_tools", lambda: mock.MagicMock(**{"invoke.return_value": response})):
        result = nodes.research_node({"messages": []}, InMemoryStore())

    assert result["research_notes"] == ["Finding: transformers scale with data."]


# --- write_node --------------------------------------------------------------

def test_write_node_returns_the_draft_and_appends_the_message() -> None:
    response = AIMessage(content="# Chapter 1\nDraft body.")
    stub = _model(invoke_result=response)

    with mock.patch.object(nodes, "get_model", lambda: stub):
        result = nodes.write_node({"research_question": "Q?"}, InMemoryStore())

    assert result["draft"] == "# Chapter 1\nDraft body."
    assert result["messages"] == [response]


def test_write_node_puts_research_notes_and_reviewer_feedback_in_the_prompt() -> None:
    stub = _model(invoke_result=AIMessage(content="draft"))
    state = {
        "research_question": "Q?",
        "outline": ["Section A"],
        "research_notes": ["NOTE-SENTINEL"],
        "review_feedback": ["FEEDBACK-SENTINEL"],
    }

    with mock.patch.object(nodes, "get_model", lambda: stub):
        nodes.write_node(state, InMemoryStore())

    system_prompt = stub.invoke.call_args[0][0][0].content
    assert "NOTE-SENTINEL" in system_prompt
    assert "FEEDBACK-SENTINEL" in system_prompt
    assert "Section A" in system_prompt


# --- reviewer_node -----------------------------------------------------------

def test_reviewer_approval_sends_the_draft_to_the_human_gate() -> None:
    review = ReviewDecision(next_step="approved", issues=[], suggestions=[])

    with mock.patch.object(nodes, "get_model", lambda: _model(structured_result=review)):
        command = nodes.reviewer_node({"draft": "d", "revision_count": 0})

    assert command.goto == "human_approval_node"
    assert command.update["review_status"] == "ai_approved"
    assert command.update["revision_count"] == 1


def test_reviewer_routes_factual_gaps_back_to_research() -> None:
    review = ReviewDecision(next_step="needs_more_research", issues=["no data"], suggestions=["find a benchmark"])

    with mock.patch.object(nodes, "get_model", lambda: _model(structured_result=review)):
        command = nodes.reviewer_node({"draft": "d", "revision_count": 0})

    assert command.goto == "research_node"
    assert command.update["review_status"] == "needs_more_research"
    assert "[ACADEMIC ISSUE] no data" in command.update["review_feedback"]
    assert "[ACADEMIC SUGGESTION] find a benchmark" in command.update["review_feedback"]


@pytest.mark.parametrize("revision_count", [0, 1])
def test_reviewer_routes_style_problems_back_to_writing(revision_count: int) -> None:
    # Both counts below the cap, so the force-approve must NOT fire yet. Together with
    # the revision_count=2 case below this pins the threshold from both sides: lowering
    # the cap breaks this test, raising it breaks the other.
    review = ReviewDecision(next_step="needs_revision", issues=["tone"], suggestions=[])

    with mock.patch.object(nodes, "get_model", lambda: _model(structured_result=review)):
        command = nodes.reviewer_node({"draft": "d", "revision_count": revision_count})

    assert command.goto == "write_node"
    assert command.update["review_status"] == "needs_revision"


def test_reviewer_force_approves_on_the_third_revision_to_guarantee_termination() -> None:
    # The only thing stopping write -> review -> write from looping forever. If this
    # test fails, the graph can spin indefinitely on a draft the reviewer never likes.
    review = ReviewDecision(next_step="needs_revision", issues=["still bad"], suggestions=[])

    with mock.patch.object(nodes, "get_model", lambda: _model(structured_result=review)):
        command = nodes.reviewer_node({"draft": "d", "revision_count": 2})

    assert command.goto == "human_approval_node"
    assert command.update["review_status"] == "max_revisions_reached"
    assert command.update["revision_count"] == 3


# --- human_approval_node -----------------------------------------------------

def test_human_approval_ends_the_graph_and_records_a_conclusion() -> None:
    store = InMemoryStore()
    state = {"draft": "d", "research_question": "Q?", "research_notes": ["evidence"]}

    with mock.patch.object(nodes, "interrupt", return_value="approve"):
        command = nodes.human_approval_node(state, store)

    assert command.goto == END
    assert command.update["review_status"] == "final_approved"
    saved = store.search(("thesis_memory", "default_user", "research_conclusion"))
    assert len(saved) == 1
    assert saved[0].value["content"] == "evidence"


def test_human_research_override_routes_back_to_research() -> None:
    with mock.patch.object(nodes, "interrupt", return_value="research: check the 2024 benchmarks"):
        command = nodes.human_approval_node({"draft": "d"}, InMemoryStore())

    assert command.goto == "research_node"
    assert command.update["review_feedback"] == ["[HUMAN FEEDBACK - RESEARCH]: check the 2024 benchmarks"]
    assert command.update["revision_count"] == 0


def test_human_revise_command_routes_back_to_writing() -> None:
    with mock.patch.object(nodes, "interrupt", return_value="revise: tighten the intro"):
        command = nodes.human_approval_node({"draft": "d"}, InMemoryStore())

    assert command.goto == "write_node"
    assert command.update["review_feedback"] == ["[HUMAN FEEDBACK - WRITING]: tighten the intro"]


def test_unrecognised_human_input_is_treated_as_revision_feedback() -> None:
    with mock.patch.object(nodes, "interrupt", return_value="make it shorter"):
        command = nodes.human_approval_node({"draft": "d"}, InMemoryStore())

    assert command.goto == "write_node"
    assert command.update["review_feedback"] == ["[HUMAN FEEDBACK - WRITING]: make it shorter"]


def test_remember_command_persists_a_durable_memory_and_returns_to_writing() -> None:
    store = InMemoryStore()
    item = DurableMemoryItem(
        category="methodology_decision",
        title="Enforce ModernBERT",
        content="Reject Word2Vec baselines.",
        status="enforced",
    )
    stub = _model(structured_result=item)

    with mock.patch.object(nodes, "get_model", lambda: stub), \
         mock.patch.object(nodes, "interrupt", return_value="remember: always use ModernBERT"):
        command = nodes.human_approval_node({"draft": "d"}, store)

    assert command.goto == "write_node"
    assert command.update["revision_count"] == 0
    saved = store.search(("thesis_memory", "default_user", "methodology_decision"))
    assert len(saved) == 1
    assert saved[0].value["title"] == "Enforce ModernBERT"
