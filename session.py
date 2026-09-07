"""Which thread to resume, and which thesis project that thread belongs to.

Both live under one namespace because this is a single-operator tool with one active
project at a time: "resume" means "carry on with the one ongoing thesis", not "pick from
several". The distinction that matters is thread vs project - a new thread is a fresh
conversation about the same thesis, a new project is a different thesis entirely.
"""

import uuid
from datetime import datetime, timezone

from langgraph.store.base import BaseStore

_NAMESPACE = ("thesis_workspace", "active_project")
_KEY = "active_thread_id"

# Read by proposal_node too. Defined here so the archive below cannot end up moving a
# different key from the one the graph actually writes.
PROPOSAL_NAMESPACE = _NAMESPACE
PROPOSAL_KEY = "current_proposal"
_ARCHIVE_PREFIX = "archived_proposal_"


def resolve_thread_id(store: BaseStore, start_new: bool = False) -> tuple[str, bool]:
    if not start_new:
        saved = store.get(_NAMESPACE, _KEY)
        if saved:
            return saved.value["thread_id"], True

    new_id = str(uuid.uuid4())
    store.put(_NAMESPACE, _KEY, {"thread_id": new_id})
    return new_id, False


def archive_active_proposal(store: BaseStore) -> str | None:
    """Move the active proposal aside so the next run proposes a genuinely new thesis.

    Archived rather than deleted: the proposal is the product of a real conversation and
    nothing here is worth destroying to save a row. Returns the archive key, or None if
    there was no active proposal to move.
    """
    current = store.get(PROPOSAL_NAMESPACE, PROPOSAL_KEY)
    if current is None:
        return None

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_key = f"{_ARCHIVE_PREFIX}{stamp}"

    # The stamp is only second-granular, and restoring archives the outgoing project
    # immediately after reading the incoming one - so two archives in the same second are
    # entirely reachable, and a collision would silently overwrite a saved thesis.
    suffix = 2
    while store.get(PROPOSAL_NAMESPACE, archive_key) is not None:
        archive_key = f"{_ARCHIVE_PREFIX}{stamp}-{suffix}"
        suffix += 1

    store.put(PROPOSAL_NAMESPACE, archive_key, current.value)
    store.delete(PROPOSAL_NAMESPACE, PROPOSAL_KEY)
    return archive_key


def list_archived_proposals(store: BaseStore) -> list[dict]:
    """Archived proposals, newest first.

    archive_active_proposal has always written these; until now nothing read them back,
    so a project could be set aside and never recovered.
    """
    found = []
    for record in store.search(PROPOSAL_NAMESPACE, limit=100):
        if record.key.startswith(_ARCHIVE_PREFIX):
            found.append({"key": record.key, **record.value})
    return sorted(found, key=lambda item: item["key"], reverse=True)


def restore_archived_proposal(store: BaseStore, archive_key: str) -> dict | None:
    """Make an archived proposal active again, archiving whatever is active first.

    Returns the restored proposal, or None if the key is unknown. Swapping rather than
    overwriting means restoring is itself reversible.
    """
    archived = store.get(PROPOSAL_NAMESPACE, archive_key)
    if archived is None or not archive_key.startswith(_ARCHIVE_PREFIX):
        return None

    # Read the value out and drop the source key *before* archiving the outgoing project,
    # so there is no ordering in which this delete can remove the archive we just wrote.
    value = archived.value
    store.delete(PROPOSAL_NAMESPACE, archive_key)
    archive_active_proposal(store)
    store.put(PROPOSAL_NAMESPACE, PROPOSAL_KEY, value)
    return value
