"""Managing what the assistant remembers, and which thesis it is working on.

Two gaps this covers. Durable memory was write-only from the operator's side: `remember:`
added constraints that steer every future generation and nothing could list or remove one,
so a single careless rule was permanent short of opening psql. And archived projects were
written by `--new-project` and never read back, so setting a thesis aside meant losing it.

Also covers the proposal gate, which moves the human decision to *before* research spends
anything - previously the plan was first seen bundled inside a finished draft.
"""

from unittest import mock

import pytest
from langgraph.store.memory import InMemoryStore

import memory
import nodes
import session
from schemas import DurableMemoryItem, SupervisorDecision, ThesisProposal

_PROPOSAL = {"thesis_topic": "T", "research_question": "Q?", "outline": ["I", "II"]}


def _model(structured_result=None) -> mock.MagicMock:
    m = mock.MagicMock(name="model")
    m.with_structured_output.return_value.invoke.return_value = structured_result
    return m


def _seed(store: InMemoryStore, count: int) -> None:
    for i in range(count):
        memory.save_durable_memory(
            store,
            DurableMemoryItem(
                category="constraint", title=f"rule-{i}", content=f"content-{i}", status="enforced"
            ),
        )


# --- listing and forgetting constraints -------------------------------------------

def test_listing_returns_every_constraint_with_its_key() -> None:
    store = InMemoryStore()
    _seed(store, 3)

    items = memory.list_durable_memories(store)

    assert [i["title"] for i in items] == ["rule-0", "rule-1", "rule-2"]
    # The key and namespace are what make deletion possible at all.
    assert all(i["key"] and i["namespace"] for i in items)


def test_listing_spans_categories() -> None:
    store = InMemoryStore()
    memory.save_durable_memory(store, DurableMemoryItem(
        category="constraint", title="a", content="x", status="enforced"))
    memory.save_durable_memory(store, DurableMemoryItem(
        category="writing_preference", title="b", content="y", status="accepted"))

    assert {i["title"] for i in memory.list_durable_memories(store)} == {"a", "b"}


def test_the_listing_is_numbered_for_the_forget_command() -> None:
    store = InMemoryStore()
    _seed(store, 2)

    rendered = memory.format_memory_listing(memory.list_durable_memories(store))

    assert "[1]" in rendered and "[2]" in rendered
    assert "rule-0" in rendered


def test_empty_listing_matches_the_prompt_wording() -> None:
    assert memory.format_memory_listing([]) == memory._NO_MEMORIES


def test_forgetting_removes_exactly_one_constraint() -> None:
    store = InMemoryStore()
    _seed(store, 3)

    removed = memory.forget_durable_memory(store, 2)

    assert removed["title"] == "rule-1"
    assert [i["title"] for i in memory.list_durable_memories(store)] == ["rule-0", "rule-2"]
    # And it stops steering generations, which is the whole point.
    assert "rule-1" not in memory.get_durable_memories(store)


@pytest.mark.parametrize("index", [0, -1, 4, 99])
def test_out_of_range_indices_return_none_and_delete_nothing(index: int) -> None:
    store = InMemoryStore()
    _seed(store, 3)

    assert memory.forget_durable_memory(store, index) is None
    assert len(memory.list_durable_memories(store)) == 3


# --- the gate commands that expose them --------------------------------------------

def test_memories_command_lists_and_returns_to_the_gate() -> None:
    store = InMemoryStore()
    _seed(store, 2)

    with mock.patch.object(nodes, "interrupt", return_value="memories"):
        command = nodes.human_approval_node({"draft": "d"}, store)

    assert command.goto == "human_approval_node"
    assert "rule-0" in command.update["review_feedback"][0]


def test_forget_command_removes_the_numbered_constraint() -> None:
    store = InMemoryStore()
    _seed(store, 2)

    with mock.patch.object(nodes, "interrupt", return_value="forget: 1"):
        command = nodes.human_approval_node({"draft": "d"}, store)

    assert command.goto == "human_approval_node"
    assert "[CONSTRAINT REMOVED] rule-0" in command.update["review_feedback"][0]
    assert len(memory.list_durable_memories(store)) == 1


def test_forget_with_a_bad_index_reports_rather_than_crashing() -> None:
    store = InMemoryStore()
    _seed(store, 1)

    with mock.patch.object(nodes, "interrupt", return_value="forget: 9"):
        command = nodes.human_approval_node({"draft": "d"}, store)

    assert "[FORGET FAILED]" in command.update["review_feedback"][0]
    assert len(memory.list_durable_memories(store)) == 1


def test_forget_with_non_numeric_input_reports_rather_than_crashing() -> None:
    with mock.patch.object(nodes, "interrupt", return_value="forget: the second one"):
        command = nodes.human_approval_node({"draft": "d"}, InMemoryStore())

    assert "[FORGET FAILED]" in command.update["review_feedback"][0]


# --- archiving and restoring projects ----------------------------------------------

def test_archived_projects_can_be_listed() -> None:
    store = InMemoryStore()
    store.put(session.PROPOSAL_NAMESPACE, session.PROPOSAL_KEY, _PROPOSAL)
    session.archive_active_proposal(store)

    archived = session.list_archived_proposals(store)

    assert len(archived) == 1
    assert archived[0]["thesis_topic"] == "T"
    assert archived[0]["key"].startswith("archived_proposal_")


def test_archive_keys_never_collide_within_the_same_second() -> None:
    """Restoring archives the outgoing project immediately after reading the incoming one.

    The timestamp is only second-granular, so those two writes land in the same second
    routinely - and a collision would overwrite a saved thesis.
    """
    store = InMemoryStore()
    keys = []
    for topic in ("A", "B", "C"):
        store.put(session.PROPOSAL_NAMESPACE, session.PROPOSAL_KEY, {**_PROPOSAL, "thesis_topic": topic})
        keys.append(session.archive_active_proposal(store))

    assert len(set(keys)) == 3
    assert sorted(a["thesis_topic"] for a in session.list_archived_proposals(store)) == ["A", "B", "C"]


def test_restoring_swaps_the_active_project_and_loses_nothing() -> None:
    store = InMemoryStore()
    store.put(session.PROPOSAL_NAMESPACE, session.PROPOSAL_KEY, {**_PROPOSAL, "thesis_topic": "A"})
    key_a = session.archive_active_proposal(store)
    store.put(session.PROPOSAL_NAMESPACE, session.PROPOSAL_KEY, {**_PROPOSAL, "thesis_topic": "B"})

    restored = session.restore_archived_proposal(store, key_a)

    assert restored["thesis_topic"] == "A"
    assert store.get(session.PROPOSAL_NAMESPACE, session.PROPOSAL_KEY).value["thesis_topic"] == "A"
    # B must have been archived in A's place, not discarded.
    assert [a["thesis_topic"] for a in session.list_archived_proposals(store)] == ["B"]


def test_restoring_an_unknown_key_is_a_noop() -> None:
    assert session.restore_archived_proposal(InMemoryStore(), "archived_proposal_nope") is None


def test_restore_refuses_a_key_outside_the_archive_namespace() -> None:
    # Otherwise `--restore-project current_proposal` would delete the active thesis.
    store = InMemoryStore()
    store.put(session.PROPOSAL_NAMESPACE, session.PROPOSAL_KEY, _PROPOSAL)

    assert session.restore_archived_proposal(store, session.PROPOSAL_KEY) is None
    assert store.get(session.PROPOSAL_NAMESPACE, session.PROPOSAL_KEY) is not None


# --- the proposal gate ---------------------------------------------------------------

def test_approving_the_plan_routes_onward_without_re_asking() -> None:
    stub = _model(SupervisorDecision(next_step="research_node"))

    with mock.patch.object(nodes, "get_model", lambda: stub), \
         mock.patch.object(nodes, "interrupt", return_value="approve"):
        command = nodes.proposal_approval_node({"thesis_topic": "T", "outline": ["I"]}, InMemoryStore())

    assert command.goto == "research_node"
    assert command.update["proposal_approved"] is True


def test_rejecting_the_plan_regenerates_it_with_the_feedback() -> None:
    with mock.patch.object(nodes, "interrupt", return_value="narrow it to retrieval only"):
        command = nodes.proposal_approval_node({"thesis_topic": "T", "outline": ["I"]}, InMemoryStore())

    assert command.goto == "proposal_node"
    assert command.update["proposal_feedback"] == ["narrow it to retrieval only"]
    # thesis_initialized must be cleared or proposal_node short-circuits on the old plan.
    assert command.update["thesis_initialized"] is False


def test_an_already_approved_plan_skips_the_gate_entirely() -> None:
    # Resuming a project approved in an earlier session must not re-ask.
    stub = _model(SupervisorDecision(next_step="write_node"))
    interrupt_spy = mock.MagicMock()

    with mock.patch.object(nodes, "get_model", lambda: stub), \
         mock.patch.object(nodes, "interrupt", interrupt_spy):
        command = nodes.proposal_approval_node({"proposal_approved": True}, InMemoryStore())

    assert command.goto == "write_node"
    interrupt_spy.assert_not_called()


def test_empty_input_at_the_plan_gate_re_asks() -> None:
    with mock.patch.object(nodes, "interrupt", return_value="   "):
        command = nodes.proposal_approval_node({"thesis_topic": "T"}, InMemoryStore())

    assert command.goto == "proposal_approval_node"


def test_the_gate_payload_carries_the_plan_for_display() -> None:
    captured = {}

    def capture(payload):
        captured.update(payload)
        return "approve"

    stub = _model(SupervisorDecision(next_step="research_node"))
    with mock.patch.object(nodes, "get_model", lambda: stub), \
         mock.patch.object(nodes, "interrupt", capture):
        nodes.proposal_approval_node(
            {"thesis_topic": "My Topic", "research_question": "Why?", "outline": ["A", "B"]},
            InMemoryStore(),
        )

    # `kind` is how the CLI tells the two gates apart.
    assert captured["kind"] == "proposal"
    assert captured["thesis_topic"] == "My Topic"
    assert captured["outline"] == ["A", "B"]


def test_a_reloaded_proposal_is_pre_approved() -> None:
    store = InMemoryStore()
    store.put(session.PROPOSAL_NAMESPACE, session.PROPOSAL_KEY, _PROPOSAL)

    result = nodes.proposal_node({"user_query": "anything"}, store)

    assert result["proposal_approved"] is True


def test_a_freshly_generated_proposal_is_not_approved_yet() -> None:
    stub = _model(ThesisProposal(thesis_topic="New", research_question="Q?", outline=["A"]))

    with mock.patch.object(nodes, "get_model", lambda: stub):
        result = nodes.proposal_node({"user_query": "an idea"}, InMemoryStore())

    assert result["proposal_approved"] is False


def test_regeneration_puts_the_humans_feedback_in_the_prompt() -> None:
    stub = _model(ThesisProposal(thesis_topic="Narrower", research_question="Q?", outline=["A"]))

    with mock.patch.object(nodes, "get_model", lambda: stub):
        nodes.proposal_node(
            {"user_query": "an idea", "proposal_feedback": ["FEEDBACK-SENTINEL"]},
            InMemoryStore(),
        )

    prompt = stub.with_structured_output.return_value.invoke.call_args[0][0]
    assert "FEEDBACK-SENTINEL" in prompt
