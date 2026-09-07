"""The approval-gate command language.

The old parser was an order-dependent `startswith` chain ending in an exact-match set
{"approve", "approved", "ok", "yes", "looks good"}. Typing "approve." missed that set and
fell through to "treat anything unrecognised as revision feedback", so saying yes with a
full stop silently spent a billed LLM call rewriting a draft the operator had just
accepted. That is the failure this module exists to prevent, and the reason the parser is
pure: the whole command language can be enumerated without a database or a graph.
"""

from unittest import mock

import pytest
from langgraph.graph import END
from langgraph.store.memory import InMemoryStore

import nodes
from hitl import HumanDecision, parse_human_decision


# --- approval, the one irreversible branch ------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "approve", "Approve", "APPROVE", "  approve  ",
        # The regression that motivated this module: punctuation used to mean "rewrite".
        "approve.", "Approve!", "approved.", "approved!!!", "OK.", "ok!", "Yes.",
        "yes", "y", "yep", "okay", "looks good", "Looks Good.", "lgtm", "LGTM!",
        "accept", "accepted", "ship it", "done",
    ],
)
def test_affirmatives_approve(text: str) -> None:
    assert parse_human_decision(text) == HumanDecision("approve")


@pytest.mark.parametrize(
    "text",
    [
        # Approval ends the graph and writes a durable memory, so it matches the WHOLE
        # input. Anything qualified is feedback, however affirmative it sounds.
        "yes, but rework the introduction",
        "ok so long as you add citations",
        "looks good apart from section 3",
        "approve after you fix the abstract",
        "not approved",
        "don't approve this",
    ],
)
def test_qualified_affirmatives_are_feedback_not_approval(text: str) -> None:
    decision = parse_human_decision(text)

    assert decision.action == "revise", f"{text!r} must not end the thesis"
    assert decision.payload == text


# --- explicit commands ---------------------------------------------------------

@pytest.mark.parametrize(
    ("text", "action", "payload"),
    [
        ("research: check the 2024 benchmarks", "research", "check the 2024 benchmarks"),
        ("revise: tighten the intro", "revise", "tighten the intro"),
        ("remember: always use ModernBERT", "remember", "always use ModernBERT"),
        # Case and spacing around the colon carry no meaning.
        ("RESEARCH: more data", "research", "more data"),
        ("Revise : tighten it", "revise", "tighten it"),
        ("remember:no spaces", "remember", "no spaces"),
        ("research:   padded   ", "research", "padded"),
    ],
)
def test_prefix_commands(text: str, action: str, payload: str) -> None:
    assert parse_human_decision(text) == HumanDecision(action, payload)


def test_multiline_payloads_survive() -> None:
    decision = parse_human_decision("revise: fix the intro\nand the conclusion")

    assert decision.action == "revise"
    assert decision.payload == "fix the intro\nand the conclusion"


def test_a_command_word_without_a_colon_is_just_feedback() -> None:
    # "research more sources" is an instruction, not the `research:` command.
    decision = parse_human_decision("research more sources")

    assert decision == HumanDecision("revise", "research more sources")


# --- empty and defaults --------------------------------------------------------

@pytest.mark.parametrize("text", ["", "   ", "\n\t ", None])
def test_empty_input_is_its_own_outcome(text) -> None:
    # Previously this fell through to a revision with empty feedback: a billed rewrite
    # instructed by nothing at all.
    assert parse_human_decision(text) == HumanDecision("empty")


def test_unrecognised_text_defaults_to_revision_feedback() -> None:
    assert parse_human_decision("make it shorter") == HumanDecision("revise", "make it shorter")


def test_every_action_can_describe_itself() -> None:
    # main.py echoes this back, so a silent misinterpretation is impossible.
    for action in ("approve", "research", "revise", "remember", "empty"):
        assert HumanDecision(action, "x").describe()


# --- the node honours the parse ------------------------------------------------

def test_empty_decision_re_asks_instead_of_rewriting() -> None:
    with mock.patch.object(nodes, "interrupt", return_value="   "):
        command = nodes.human_approval_node({"draft": "d"}, InMemoryStore())

    assert command.goto == "human_approval_node", "empty input must re-ask, not redraft"


def test_approve_with_a_full_stop_ends_the_thesis() -> None:
    # The end-to-end version of the headline bug: this used to trigger a rewrite.
    store = InMemoryStore()
    state = {"draft": "d", "research_question": "Q?", "research_notes": ["evidence"]}

    with mock.patch.object(nodes, "interrupt", return_value="Approve."):
        command = nodes.human_approval_node(state, store)

    assert command.goto == END
    assert command.update["review_status"] == "final_approved"
    assert len(store.search(("thesis_memory", "default_user", "research_conclusion"))) == 1


def test_qualified_yes_still_routes_to_a_revision() -> None:
    with mock.patch.object(nodes, "interrupt", return_value="yes, but shorten section 2"):
        command = nodes.human_approval_node({"draft": "d"}, InMemoryStore())

    assert command.goto == "write_node"
    assert command.update["review_feedback"] == [
        "[HUMAN FEEDBACK - WRITING]: yes, but shorten section 2"
    ]


def test_cli_echoes_its_interpretation() -> None:
    import inspect
    import main

    source = inspect.getsource(main._service_interrupts)
    assert "parse_human_decision" in source and "describe()" in source, (
        "the CLI must state how it read the input, so a fallthrough is never silent"
    )
