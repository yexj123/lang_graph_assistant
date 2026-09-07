"""Token accounting for a run.

A thesis run makes dozens of billed calls - a research turn alone loops research_node and
tool_node until the model stops asking for tools - and until now none of it was visible.
This tracks what LangChain already reports on every response.

Tokens, not currency, by default. Per-token prices differ per provider, change without
notice and would be stale in the repository within weeks; a hardcoded rate that quietly
drifts is worse than no number. Set `<PROVIDER>_INPUT_PRICE` / `<PROVIDER>_OUTPUT_PRICE`
(USD per million tokens) if you want the estimate, and it is labelled an estimate.

Pure: no config, no langchain. `record()` takes any object and reads `usage_metadata`
defensively, because not every provider populates it and a missing count must never break
a turn.
"""

import os
from dataclasses import dataclass, field


@dataclass
class Usage:
    """Running totals for one CLI session."""

    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    by_model: dict[str, int] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def record(self, response: object, model_name: str = "") -> None:
        """Accumulate one model response. Silently ignores anything without usage data."""
        metadata = getattr(response, "usage_metadata", None)
        if not isinstance(metadata, dict):
            return

        self.input_tokens += int(metadata.get("input_tokens", 0) or 0)
        self.output_tokens += int(metadata.get("output_tokens", 0) or 0)
        self.calls += 1

        name = model_name or getattr(response, "response_metadata", {}).get("model_name", "unknown")
        total = int(metadata.get("total_tokens", 0) or 0) or (
            int(metadata.get("input_tokens", 0) or 0) + int(metadata.get("output_tokens", 0) or 0)
        )
        self.by_model[name] = self.by_model.get(name, 0) + total

    def estimated_cost(self, provider_name: str) -> float | None:
        """USD estimate, or None when no prices are configured for this provider."""
        prefix = provider_name.upper()
        try:
            in_price = float(os.environ[f"{prefix}_INPUT_PRICE"])
            out_price = float(os.environ[f"{prefix}_OUTPUT_PRICE"])
        except (KeyError, ValueError):
            return None
        return (self.input_tokens * in_price + self.output_tokens * out_price) / 1_000_000

    def summary(self, provider_name: str = "") -> str:
        """One line for the CLI after each turn."""
        if not self.calls:
            return "No model calls recorded yet."

        line = (
            f"{self.calls} call(s), {self.total_tokens:,} tokens "
            f"({self.input_tokens:,} in / {self.output_tokens:,} out)"
        )
        cost = self.estimated_cost(provider_name) if provider_name else None
        if cost is not None:
            line += f" — est. ${cost:.4f}"
        return line

    def detail(self) -> str:
        """Per-model breakdown for `--status` and end-of-session reporting."""
        if not self.by_model:
            return self.summary()
        lines = [self.summary()]
        for name, total in sorted(self.by_model.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {name}: {total:,} tokens")
        return "\n".join(lines)
