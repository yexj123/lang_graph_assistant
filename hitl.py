"""Parsing for the human-in-the-loop approval gate.

Kept free of config/langchain/langgraph imports, like sandbox.py, so the protocol can be
unit-tested exhaustively without a database and so main.py can echo its interpretation
back to the operator without pulling in the graph stack.

The old parser was an order-dependent chain of `startswith` checks ending in an
exact-match set `{"approve", "approved", "ok", "yes", "looks good"}`. Two problems:

* `"approve."`, `"Approved!"` and `"OK."` missed the set and fell through to the final
  else, which treats anything unrecognised as revision feedback. Saying yes with a full
  stop silently triggered a full rewrite - a billed LLM call and a changed draft - and
  nothing told the operator their approval had been reinterpreted.
* The fallthrough was invisible. Every unrecognised string became a rewrite instruction,
  so a question like "what did the reviewer object to?" was fed back as an edit.

The rules now, in order:

1. `remember:` / `research:` / `revise:` prefixes, tolerating case and spacing.
2. Approval, matched against the *whole* normalised input with trailing punctuation
   removed. Whole-input matching is deliberate: approval is the one irreversible branch
   (it ends the graph and writes a durable memory), so `"yes, but redo the intro"` must
   never approve.
3. Empty input is its own outcome, not a rewrite with empty feedback.
4. Anything else is revision feedback - still the right default, but now labelled so the
   caller can say so out loud.
"""

import re
from dataclasses import dataclass
from typing import Literal

HumanAction = Literal["approve", "research", "revise", "remember", "empty"]

# Optional whitespace either side of the colon: "Revise : tighten the intro" is a command.
_COMMAND_RE = re.compile(r"^(remember|research|revise)\s*:\s*(.*)$", re.IGNORECASE | re.DOTALL)

# Whole-input matches only. Anything longer is feedback, however affirmative it sounds.
_APPROVAL_WORDS = frozenset(
    {
        "approve", "approved", "approve it",
        "ok", "okay", "k",
        "yes", "y", "yep", "yeah",
        "looks good", "lgtm",
        "accept", "accepted",
        "ship it", "done",
    }
)

# Trailing punctuation and surrounding quotes carry no meaning here.
_TRIM = " \t.!,;:'\"()"


@dataclass(frozen=True)
class HumanDecision:
    """What the operator asked for at the approval gate."""

    action: HumanAction
    payload: str = ""

    def describe(self) -> str:
        """One line the CLI can echo, so an interpretation is never silent."""
        return {
            "approve": "approving the draft and finalising this thesis",
            "research": "sending it back for more research",
            "revise": "sending it back for a writing revision",
            "remember": "saving a durable constraint, then revising",
            "empty": "no decision entered",
        }[self.action]


def parse_human_decision(raw: object) -> HumanDecision:
    """Interpret the operator's free-text reply at the approval gate."""
    text = "" if raw is None else str(raw).strip()
    if not text:
        return HumanDecision("empty")

    command = _COMMAND_RE.match(text)
    if command:
        action = command.group(1).lower()
        return HumanDecision(action, command.group(2).strip())  # type: ignore[arg-type]

    if text.lower().strip(_TRIM) in _APPROVAL_WORDS:
        return HumanDecision("approve")

    return HumanDecision("revise", text)
