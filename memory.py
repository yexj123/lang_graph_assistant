import uuid
from typing import get_args

from langgraph.store.base import BaseStore

from schemas import DecisionCategory, DurableMemoryItem

# Derived from the schema rather than retyped. A hardcoded copy silently stops reading
# any category added to DecisionCategory later - the memories would be written and then
# never injected into a prompt again, with nothing to indicate it.
CATEGORIES: tuple[str, ...] = get_args(DecisionCategory)

# BaseStore.search defaults to limit=10. Relying on that default silently dropped every
# constraint past the tenth in a category, so a rule the user explicitly saved just
# stopped being enforced. Page through instead, with an explicit limit.
_PAGE_SIZE = 100

# Constraints are injected into every research, write and approval prompt, so the block
# cannot grow without bound. When it does hit the cap, say so in the text itself rather
# than truncating in silence - that is the whole bug being fixed here.
MAX_INJECTED_MEMORIES = 60

_NO_MEMORIES = "No durable decisions recorded yet."


def save_durable_memory(
    store: BaseStore, item: DurableMemoryItem, user_id: str = "default_user"
) -> None:
    """Persists a single curated memory item into PostgresStore."""
    namespace = ("thesis_memory", user_id, item.category)
    store.put(
        namespace=namespace,
        key=str(uuid.uuid4()),
        value=item.model_dump()
    )


def _search_all(store: BaseStore, namespace: tuple[str, ...]) -> list:
    """Every item in a namespace, not just the store's default first page."""
    items: list = []
    offset = 0
    while True:
        page = store.search(namespace, limit=_PAGE_SIZE, offset=offset)
        items.extend(page)
        if len(page) < _PAGE_SIZE:
            return items
        offset += _PAGE_SIZE


def count_durable_memories(store: BaseStore, user_id: str = "default_user") -> int:
    """Total saved constraints across all categories, including any beyond the cap."""
    return sum(
        len(_search_all(store, ("thesis_memory", user_id, category)))
        for category in CATEGORIES
    )


def get_durable_memories(store: BaseStore, user_id: str = "default_user") -> str:
    """Fetches all stored thesis rules, constraints, and conclusions formatted for prompts."""
    records = []
    for category in CATEGORIES:
        records.extend(_search_all(store, ("thesis_memory", user_id, category)))

    if not records:
        return _NO_MEMORIES

    shown = records[:MAX_INJECTED_MEMORIES]
    lines = [
        f"• [{r.value['category'].upper()} - {r.value['status'].upper()}] "
        f"{r.value['title']}: {r.value['content']}"
        for r in shown
    ]

    hidden = len(records) - len(shown)
    if hidden:
        lines.append(
            f"... and {hidden} further constraint(s) not shown - {len(records)} are saved but only "
            f"{MAX_INJECTED_MEMORIES} fit in the prompt. Prune them, or raise "
            f"MAX_INJECTED_MEMORIES in memory.py, if the omitted ones still matter."
        )

    return "\n".join(lines)
