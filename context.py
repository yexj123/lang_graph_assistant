"""Keep the prompt from growing without bound across a long thesis project.

`messages` and `research_notes` are both `Annotated[..., operator.add]`, so every research
turn, every tool result and every full draft is appended and never removed. In a system
explicitly designed to loop research -> write -> review -> research over weeks, that grows
monotonically until it exceeds the context window - and the failure arrives as an API
error mid-turn, after the run has already been paid for.

Two rules, both conservative:

* Messages are trimmed with LangChain's `trim_messages`, not by slicing. A naive
  `messages[-20:]` can orphan a ToolMessage from the AIMessage that requested it, which
  every provider rejects. `start_on="human"` and `end_on=(...)` are what keep tool-call
  pairs intact.
* Notes are kept newest-first within a character budget, and the drop is *announced* in
  the text rather than silent - the same rule the durable-memory cap follows.
"""

from langchain_core.messages import BaseMessage, trim_messages

# Generous: large enough that a normal session never trims, small enough that a runaway
# tool loop cannot push the prompt past a typical context window.
MAX_HISTORY_TOKENS = 60_000
MAX_NOTES_CHARS = 40_000

_CHARS_PER_TOKEN = 4  # rough, and only used to decide when to trim, never to bill


def approx_tokens(messages: list[BaseMessage]) -> int:
    """Cheap length estimate. Deliberately not tiktoken: this decides whether to trim,
    and paying a real tokenizer pass on every node call to answer that is not worth it."""
    return sum(len(str(getattr(m, "content", ""))) for m in messages) // _CHARS_PER_TOKEN


def trim_history(
    messages: list[BaseMessage], max_tokens: int = MAX_HISTORY_TOKENS
) -> list[BaseMessage]:
    """The most recent messages that fit, with tool-call pairing preserved.

    Returns the input unchanged when it already fits, so the common case costs one
    length estimate and nothing else.
    """
    if not messages or approx_tokens(messages) <= max_tokens:
        return messages

    trimmed = trim_messages(
        messages,
        max_tokens=max_tokens,
        strategy="last",
        # Character-based counting, matching approx_tokens: consistent with the check
        # above, and free of a tokenizer dependency that would differ per provider.
        token_counter=lambda msgs: approx_tokens(list(msgs)),
        # A conversation may not begin with a tool result, and may not end mid-tool-call.
        start_on="human",
        end_on=("human", "ai", "tool"),
        include_system=True,
        allow_partial=False,
    )
    # trim_messages can return nothing if a single message exceeds the budget; keeping the
    # last message is worse than useless only if it is empty, and better than sending [].
    return trimmed or messages[-1:]


def condense_notes(notes: list[str], max_chars: int = MAX_NOTES_CHARS) -> list[str]:
    """Newest notes that fit in `max_chars`, with an explicit marker if any were dropped.

    Newest-first because reviewer feedback drives the later notes, so they are the ones
    the current draft has to answer to.
    """
    if not notes:
        return []

    kept: list[str] = []
    budget = max_chars
    for note in reversed(notes):
        cost = len(note) + 1
        if cost > budget:
            break
        kept.append(note)
        budget -= cost

    kept.reverse()
    dropped = len(notes) - len(kept)
    if dropped:
        # Announced, not silent - the writer must know its evidence is partial.
        kept.insert(
            0,
            f"[{dropped} earlier research note(s) omitted to fit the context window; "
            f"{len(notes)} were gathered in total.]",
        )
    return kept
