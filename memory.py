import uuid
from langgraph.store.base import BaseStore
from schemas import DurableMemoryItem, DecisionCategory

def save_durable_memory(store: BaseStore, item: DurableMemoryItem, user_id: str = "default_user"):
    """Persists a single curated memory item into PostgresStore."""
    namespace = ("thesis_memory", user_id, item.category)
    store.put(
        namespace=namespace,
        key=str(uuid.uuid4()),
        value=item.model_dump()
    )

def get_durable_memories(store: BaseStore, user_id: str = "default_user") -> str:
    """Fetches all stored thesis rules, constraints, and conclusions formatted for prompts."""
    categories = [
        "thesis_decision",
        "research_direction",
        "research_conclusion",
        "writing_preference",
        "supervisor_feedback",
        "constraint",
        "methodology_decision",
    ]

    formatted_items = []
    for cat in categories:
        records = store.search(("thesis_memory", user_id, cat))
        for r in records:
            data = r.value
            formatted_items.append(
                f"• [{data['category'].upper()} - {data['status'].upper()}] {data['title']}: {data['content']}"
            )

    return "\n".join(formatted_items) if formatted_items else "No durable decisions recorded yet."