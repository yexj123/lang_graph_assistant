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

import providers
from providers import Provider

load_dotenv()

# Cheap module-level constants: reading env vars performs no I/O and cannot fail.
#
# Model selection is provider-agnostic. Set LLM_MODEL (or the legacy OPENAI_MODEL) to any
# id the registry in providers.py recognises - `gpt-4o`, `claude-sonnet-5`,
# `deepseek-chat` - and the provider is inferred from the prefix. LLM_PROVIDER overrides
# that inference when a model id is ambiguous or newly released.
MODEL_NAME, MODEL_PROVIDER = providers.resolve(
    os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL", ""),
    os.getenv("LLM_PROVIDER", ""),
)

# Embeddings stay on OpenAI regardless of the chat provider: the pgvector collection is
# built with a specific embedding model and its dimensionality, so switching chat models
# must not silently invalidate the index.
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# The RAG evaluation's judge. It must differ from MODEL_NAME: an LLM grading output it
# produced itself has a documented self-preference bias that inflates exactly the two
# metrics the eval leans on (faithfulness and answer relevancy). Independence is the
# requirement here, not capability - point this at whichever distinct model you trust,
# and a *different provider* is the strongest form of independence available.
JUDGE_MODEL_NAME, JUDGE_PROVIDER = providers.resolve(
    os.getenv("JUDGE_MODEL", "gpt-4o-mini"),
    os.getenv("JUDGE_PROVIDER", ""),
)
DB_URI = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/thesis_db")
VECTOR_DB_URI = DB_URI.replace("postgresql://", "postgresql+psycopg://")

VECTOR_COLLECTION_NAME = "thesis_literature"


def _build_chat_model(model_name: str, provider: Provider, temperature: float = 0):
    """Construct a chat client for one provider.

    The only place in the codebase that knows provider-specific client classes. OpenAI-
    compatible providers (DeepSeek) reuse ChatOpenAI with a different base_url and so
    need no extra dependency; Anthropic needs langchain-anthropic, which is imported
    lazily so that a user who only wants GPT never has to install it.
    """
    if provider.name == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise RuntimeError(
                f"{model_name!r} is an Anthropic model, which needs the langchain-anthropic "
                f"package:\n    pip install langchain-anthropic\n"
                f"Or set LLM_PROVIDER/OPENAI_MODEL to a provider you have installed."
            ) from exc
        return ChatAnthropic(model=model_name, temperature=temperature)

    kwargs = {"model": model_name, "temperature": temperature}
    if provider.base_url:
        kwargs["base_url"] = provider.base_url
        # ChatOpenAI reads OPENAI_API_KEY implicitly; a compatible provider keeps its own.
        key = os.getenv(provider.api_key_env)
        if not key:
            raise RuntimeError(
                f"{model_name!r} needs {provider.api_key_env} to be set "
                f"(provider: {provider.name}, endpoint: {provider.base_url})."
            )
        kwargs["api_key"] = key
    return ChatOpenAI(**kwargs)


@lru_cache(maxsize=1)
def get_model():
    """The shared chat model, for whichever provider MODEL_NAME/LLM_PROVIDER select."""
    return _build_chat_model(MODEL_NAME, MODEL_PROVIDER)


@lru_cache(maxsize=1)
def get_judge_model():
    """The evaluation judge. Deliberately a separate client from get_model()."""
    return _build_chat_model(JUDGE_MODEL_NAME, JUDGE_PROVIDER)


@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    """Always OpenAI: the pgvector collection is tied to this model's dimensionality."""
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

    if "api_key" in lowered or "credential" in lowered or "langchain-anthropic" in lowered:
        # Name the key the *selected provider* needs, not always OpenAI's - with three
        # providers in play, "set OPENAI_API_KEY" is actively misleading advice.
        hint = (
            f"{MODEL_PROVIDER.api_key_env} is not set, and model {MODEL_NAME!r} is served by "
            f"{MODEL_PROVIDER.name}. Create a .env file in the repo root containing\n"
            f"  {MODEL_PROVIDER.api_key_env}=...\n"
            f"or point LLM_MODEL at a provider you have configured:\n\n"
            f"{providers.describe_available()}"
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
