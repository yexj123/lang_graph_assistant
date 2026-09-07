"""Bounding the prompt, and making the bill visible.

`messages` and `research_notes` are append-only reducers in a system designed to loop, so
without trimming the prompt grows until an API rejects it - mid-turn, after the run has
already been paid for. And until `usage.py`, dozens of billed calls per session produced
no number at all.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from context import approx_tokens, condense_notes, trim_history
from usage import Usage


def _tool_exchange(query: str) -> list:
    """A realistic research turn: ask, request a tool, receive the result."""
    call_id = f"call_{abs(hash(query)) % 10000}"
    return [
        HumanMessage(content=query),
        AIMessage(content="", tool_calls=[{"name": "web_search", "args": {"q": query}, "id": call_id}]),
        ToolMessage(content=f"results for {query}", tool_call_id=call_id),
    ]


# --- message trimming ------------------------------------------------------------

def test_short_history_is_returned_unchanged() -> None:
    # The common case must cost one length estimate and nothing else.
    messages = [HumanMessage(content="hi"), AIMessage(content="hello")]

    assert trim_history(messages) is messages


def test_empty_history_is_handled() -> None:
    assert trim_history([]) == []


def test_long_history_is_trimmed() -> None:
    messages = [HumanMessage(content="x" * 4000) for _ in range(60)]

    trimmed = trim_history(messages, max_tokens=2_000)

    assert len(trimmed) < len(messages)
    assert approx_tokens(trimmed) <= 2_000


def test_trimming_never_orphans_a_tool_result() -> None:
    """The reason this uses trim_messages rather than a slice.

    A bare `messages[-N:]` can start the history on a ToolMessage whose requesting
    AIMessage was dropped, which every provider rejects with a hard error.
    """
    messages = [SystemMessage(content="sys")]
    for i in range(30):
        messages += _tool_exchange(f"query {i} " + "padding " * 200)

    trimmed = trim_history(messages, max_tokens=1_500)

    non_system = [m for m in trimmed if not isinstance(m, SystemMessage)]
    assert non_system, "trimming must not empty the history"
    assert not isinstance(non_system[0], ToolMessage), "history starts on an orphaned tool result"

    # Every tool result still has its requesting call somewhere before it.
    open_calls: set[str] = set()
    for message in trimmed:
        for call in getattr(message, "tool_calls", None) or []:
            open_calls.add(call["id"])
        if isinstance(message, ToolMessage):
            assert message.tool_call_id in open_calls, "orphaned ToolMessage"


def test_trimming_keeps_the_most_recent_exchange() -> None:
    messages = [HumanMessage(content=f"message {i} " + "x" * 2000) for i in range(40)]

    trimmed = trim_history(messages, max_tokens=1_000)

    assert trimmed[-1].content == messages[-1].content


def test_an_oversized_single_message_still_yields_something() -> None:
    # Returning [] here would send a prompt with no user content at all.
    messages = [HumanMessage(content="x" * 100_000)]

    assert trim_history(messages, max_tokens=10) != []


# --- note condensing --------------------------------------------------------------

def test_notes_within_budget_are_untouched() -> None:
    notes = ["short one", "short two"]

    assert condense_notes(notes) == notes


def test_empty_notes() -> None:
    assert condense_notes([]) == []


def test_older_notes_are_dropped_newest_first() -> None:
    notes = [f"note-{i} " + "x" * 500 for i in range(20)]

    kept = condense_notes(notes, max_chars=2_000)

    assert notes[-1] in kept, "the newest note must survive"
    assert notes[0] not in kept, "the oldest should be the first dropped"


def test_dropping_notes_is_announced_not_silent() -> None:
    # Same rule the durable-memory cap follows: the writer must know its evidence is
    # partial, otherwise it drafts confidently from a truncated record.
    notes = [f"note-{i} " + "x" * 500 for i in range(20)]

    kept = condense_notes(notes, max_chars=2_000)

    assert any("omitted to fit the context window" in note for note in kept)
    assert any("20 were gathered in total" in note for note in kept)


def test_no_notice_when_nothing_was_dropped() -> None:
    assert not any("omitted" in n for n in condense_notes(["a", "b"], max_chars=10_000))


# --- usage accounting ---------------------------------------------------------------

class _Response:
    def __init__(self, inp: int, out: int, model: str = "gpt-4o") -> None:
        self.usage_metadata = {"input_tokens": inp, "output_tokens": out, "total_tokens": inp + out}
        self.response_metadata = {"model_name": model}


def test_nothing_recorded_yet() -> None:
    assert "No model calls" in Usage().summary()


def test_totals_accumulate_across_calls() -> None:
    usage = Usage()
    usage.record(_Response(100, 20))
    usage.record(_Response(300, 50))

    assert usage.calls == 2
    assert usage.input_tokens == 400
    assert usage.output_tokens == 70
    assert usage.total_tokens == 470


def test_responses_without_usage_metadata_are_ignored_not_fatal() -> None:
    # Not every provider populates usage_metadata; a missing count must never break a turn.
    usage = Usage()
    usage.record(object())
    usage.record(None)
    usage.record(_Response(10, 5))

    assert usage.calls == 1


def test_tokens_are_attributed_per_model() -> None:
    usage = Usage()
    usage.record(_Response(100, 20), "gpt-4o")
    usage.record(_Response(200, 30), "claude-sonnet-5")

    assert usage.by_model == {"gpt-4o": 120, "claude-sonnet-5": 230}
    assert "claude-sonnet-5" in usage.detail()


def test_cost_is_none_unless_prices_are_configured() -> None:
    # Hardcoded per-token rates drift and would be stale in the repo within weeks, so the
    # default output is tokens - a number that cannot silently become wrong.
    usage = Usage()
    usage.record(_Response(1_000_000, 1_000_000))

    assert usage.estimated_cost("openai") is None
    assert "est. $" not in usage.summary("openai")


def test_cost_is_estimated_when_prices_are_supplied(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_INPUT_PRICE", "2.50")
    monkeypatch.setenv("OPENAI_OUTPUT_PRICE", "10.00")
    usage = Usage()
    usage.record(_Response(1_000_000, 1_000_000))

    assert usage.estimated_cost("openai") == 12.50
    assert "est. $12.5000" in usage.summary("openai")


def test_malformed_prices_do_not_raise(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_INPUT_PRICE", "not-a-number")
    monkeypatch.setenv("OPENAI_OUTPUT_PRICE", "10")
    usage = Usage()
    usage.record(_Response(10, 10))

    assert usage.estimated_cost("openai") is None
