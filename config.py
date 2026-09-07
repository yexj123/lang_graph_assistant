"""Dependency-injection root: every shared client is constructed here, exactly once.

Accessors are lazy on purpose. Constructing these eagerly at module scope made
`import config` a blocking side effect — `ChatOpenAI` raises without OPENAI_API_KEY
and `PGVector` opens a Postgres connection in its constructor — so importing any
module in this project transitively required a live database and a valid key.
`python main.py --help` crashed after ~25s for that reason, and nothing could be
unit-tested without stubbing this module out.

Nothing below touches the network until the corresponding get_* is first called.
"""

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from psycopg_pool import ConnectionPool

load_dotenv()

# Cheap module-level constants: reading env vars performs no I/O and cannot fail.
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o")
EMBEDDING_MODEL_NAME = "text-embedding-3-small"

# The RAG evaluation's judge. It must differ from MODEL_NAME: an LLM grading output it
# produced itself has a documented self-preference bias that inflates exactly the two
# metrics the eval leans on (faithfulness and answer relevancy). Independence is the
# requirement here, not capability - point this at whichever distinct model you trust.
JUDGE_MODEL_NAME = os.getenv("JUDGE_MODEL", "gpt-4o-mini")
DB_URI = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/thesis_db")
VECTOR_DB_URI = DB_URI.replace("postgresql://", "postgresql+psycopg://")

VECTOR_COLLECTION_NAME = "thesis_literature"


@lru_cache(maxsize=1)
def get_model() -> ChatOpenAI:
    """The shared chat model. Raises if OPENAI_API_KEY is unset."""
    return ChatOpenAI(model=MODEL_NAME, temperature=0)


@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=1)
def get_pool() -> ConnectionPool:
    """Connection pool backing both persistence layers (checkpointer and store)."""
    return ConnectionPool(conninfo=DB_URI, max_size=20, kwargs={"autocommit": True})


@lru_cache(maxsize=1)
def get_vector_store() -> PGVector:
    """pgvector-backed literature index.

    Deferred, but still blocking on first call: PGVector connects in its constructor
    to create the `vector` extension. connect_timeout applies per connection attempt
    and SQLAlchemy makes several, so an unreachable server takes 20-50s to fail, not
    the ~5s the setting suggests.
    """
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=VECTOR_COLLECTION_NAME,
        connection=VECTOR_DB_URI,
        engine_args={"connect_args": {"connect_timeout": 5}},
    )


@lru_cache(maxsize=1)
def get_checkpointer() -> PostgresSaver:
    """Per-thread graph state, keyed by thread_id."""
    return PostgresSaver(get_pool())


@lru_cache(maxsize=1)
def get_store() -> PostgresStore:
    """Cross-session long-term memory, keyed by namespace tuple."""
    return PostgresStore(get_pool())


def startup_error_message(exc: Exception) -> str:
    """Turn a driver-level failure into something the operator can act on.

    The original exception text is always included - this narrows the diagnosis,
    it never hides it.
    """
    text = f"{type(exc).__name__}: {exc}"
    lowered = text.lower()

    if "api_key" in lowered or "openai_api_key" in lowered or "credential" in lowered:
        hint = (
            "OPENAI_API_KEY is not set. Create a .env file in the repo root containing\n"
            "  OPENAI_API_KEY=sk-...\n"
            "or export it in your shell."
        )
    elif "connect" in lowered or "operational" in lowered or "timeout" in lowered:
        hint = (
            f"Cannot reach Postgres at {DB_URI}\n"
            "Check that the server is running, the database exists, and that the\n"
            "pgvector extension is available. Override the DSN with DATABASE_URL in .env."
        )
    else:
        hint = "Startup failed before the assistant could begin."

    return f"\n[startup error] {hint}\n\nOriginal error: {text}"


def init_db() -> None:
    """Create the checkpointer and store tables. Call once, explicitly, at startup."""
    get_checkpointer().setup()
    get_store().setup()
