# Developer Guide

An orientation for someone about to change this codebase. For *why* things are the way they are, see
[`CLAUDE.md`](./CLAUDE.md); for what was broken and how it was fixed, see [`CHANGELOG.md`](./CHANGELOG.md).

The whole system is ~2,600 lines of Python across 19 modules and 331 tests. Every name below is real —
grep for it.

---

## 1. What each part does

### The graph

| File | Responsibility |
|---|---|
| **`state.py`** | Defines `ThesisState`, the single `TypedDict` threaded through every node. `messages` and `research_notes` are `Annotated[..., operator.add]` (append-only); every other field is overwritten by whichever node returns it. |
| **`graph.py`** | `build_graph()` registers the six nodes plus a `ToolNode`, wires the static edges, and compiles with the checkpointer and store. It is the *only* place the graph topology is declared — and it assumes `config.init_db()` has already run. |
| **`nodes.py`** | The eight callables that make every decision: `supervisor_router`, `research_router`, `proposal_node`, `proposal_approval_node`, `research_node`, `write_node`, `reviewer_node`, `human_approval_node`. This is where the product lives. |
| **`schemas.py`** | The Pydantic models every LLM decision is forced through: `ThesisProposal`, `SupervisorDecision`, `ReviewDecision`, `DurableMemoryItem`, `GeneratedCode`, plus the `DecisionCategory` Literal. No node parses LLM prose. |

### Capabilities

| File | Responsibility |
|---|---|
| **`tools.py`** | The five tools bound to the research agent — `search_thesis_literature`, `web_search`, `file_reader`, `code_generator`, `python_executor` — plus `format_citation()`, which turns chunk metadata into `filename, p.N`, and `get_model_with_tools()`. |
| **`sandbox.py`** | `run_sandboxed(code, timeout)` executes code in a child `python` process with an allowlisted environment, stdin closed, a 15s timeout and truncated output. Deliberately imports nothing from this project, so it is testable without a database. **Not a security boundary** — see the README security note. |
| **`memory.py`** | Cross-session durable constraints: `save_durable_memory()`, `get_durable_memories()` (the prompt-ready block, paginated and capped at `MAX_INJECTED_MEMORIES`) and `count_durable_memories()`. `CATEGORIES` is derived from `DecisionCategory` via `get_args`, never retyped. |
| **`session.py`** | Thread and project lifecycle: `resolve_thread_id()` decides which conversation to resume, `archive_active_proposal()` moves the current thesis aside for `--new-project`. Owns `PROPOSAL_NAMESPACE`/`PROPOSAL_KEY`, which `nodes.py` imports. |
| **`hitl.py`** | `parse_human_decision()` turns the operator's free-text reply at either gate into a `HumanDecision(action, payload)`. Pure — imports only `re`/`dataclasses`/`typing` — so both `nodes.py` and `main.py` can use it and the command language is exhaustively testable. |
| **`export.py`** | `write_draft()` writes the thesis to `drafts/<slug>-<timestamp>.<ext>`, appending a References section built from the draft's own citations. `slugify()` and `draft_filename()` handle the filename rules. |
| **`render.py`** | `render()` dispatches per format. `markdown_to_latex()` and `escape_tex()` are pure; `markdown_to_docx()` needs `python-docx`; `markdown_to_pdf()` delegates to pandoc or pdflatex via `pdf_toolchain()`. Every missing dependency raises `RenderError` naming the fix. |
| **`bibliography.py`** | `extract_citations()` parses `[source, p.N]` markers, `build_reference_list()` resolves them against `papers/manifest.json`, `to_bibtex()` emits arXiv `@misc` entries. Pure — the manifest is passed in as data. |
| **`providers.py`** | The model→provider registry. `resolve()` settles a `(model, Provider)` pair from `LLM_MODEL`/`LLM_PROVIDER`; `describe_available()` powers `--status`. Pure data, no clients. |
| **`context.py`** | `trim_history()` bounds the message list using LangChain's `trim_messages` (preserving tool-call pairing), `condense_notes()` bounds the notes and announces what it dropped. |
| **`usage.py`** | `Usage.record()` accumulates `usage_metadata` from each response; `summary()`/`detail()` report calls, tokens and an optional cost estimate. |

### Infrastructure and entry points

| File | Responsibility |
|---|---|
| **`config.py`** | The DI root. Six `@lru_cache` accessors (`get_model`, `get_embeddings`, `get_pool`, `get_vector_store`, `get_checkpointer`, `get_store`), the `init_db()` migration, and `startup_error_message()`. Every accessor is lazy: importing this module performs no I/O. |
| **`main.py`** | The CLI. `run_cli()` handles startup, session resumption and the query loop; `_run_turn()` runs one graph invocation; `_service_interrupts()` drives the human approval gate. Only `argparse` and `hitl` are imported at module scope. |
| **`ingest.py`** | `run_ingestion(rebuild)` chunks the PDFs in `papers/` and upserts them into pgvector. `chunk_id()` gives each chunk a `sha256(source, page, text)` id, which is what makes re-running idempotent. |
| **`main.py` one-shots** | `print_status()`, `print_projects()` and `restore_project()` run without entering the loop; `_stream()` prints each node as it completes; `_render_proposal_gate()` draws the plan gate. |
| **`fetch_papers.py`** | Downloads the reference corpus pinned in `papers/manifest.json` (`load_manifest`, `download`, `sha256_of`, `fetch_all`). Checksum-verified and idempotent; the PDFs themselves are gitignored. |

---

## 2. How the parts relate

### Control flow through the graph

```
                    python main.py
                          │
                    run_cli()  ── init_db() ─→ build_graph() ─→ get_pool()/get_store()
                          │
                 resolve_thread_id(store)
                          │
              ┌───────────┴────────────┐
       .next is set?              .next is empty
              │                        │
     _service_interrupts()      prompt "Enter your query:"
              │                        │
              └──────────┬─────────────┘
                         ▼
                    _run_turn() ── _stream(graph, ...) ────────┐   prints each node,
                                                               │   records usage
  ═══════════════════════ THE GRAPH ═══════════════════════════╪══════════════
                                                               ▼
   START ─→ proposal_node ─→ proposal_approval_node
                    ▲                  │ interrupt({"kind": "proposal"})
                    │                  │  ↕ THE PLAN GATE — before anything is billed
                    └── reject ────────┤
                                       │ approve → Command(goto=supervisor_router(state))
                       ┌───────────────┴───────────────┐
                       ▼                               ▼
                 research_node ──┐                 write_node
                       ▲         │                     │
                       │  [research_router]            │
                 tool_node       │                     │
                       ▲         │                     │
                       └─────────┤                     │
                                 └──→ write_node ──────┤
                                                       ▼
                                                 reviewer_node
                                                       │  Command(goto=...)
                 ┌─────────────────────────────────────┼─────────────────────┐
                 ▼                                     ▼                     ▼
          research_node                          write_node        human_approval_node
                                                                            │ interrupt(payload)
                                                                            │  ↕ THE DRAFT GATE
                                            parse_human_decision(reply) ────┤
   ┌────────┬────────┬─────┬──────────┬─────────┬────────┬──────────────────┘
   ▼        ▼        ▼     ▼          ▼         ▼        ▼
research  write    END   export   memories   forget   human_approval_node
 _node    _node                                        (empty input → re-ask;
                                                        export/memories/forget
                                                        also loop back here)
```

**Two gates, one loop.** `_service_interrupts` drives both; it tells them apart by
`payload["kind"] == "proposal"`. The plan gate exists so the human sees the topic,
question and outline *before* `research_node` spends anything — previously the plan was
first visible bundled inside a finished draft.

Static edges only get you as far as `reviewer_node`. From there **routing is invisible in `graph.py`** —
`reviewer_node` and `human_approval_node` return `Command(goto=...)`, so the real control flow is in
`nodes.py`. `reviewer_node` force-approves once `revision_count >= 3` regardless of quality; that is the
graph's only termination guarantee, and `tests/test_nodes.py` pins it from both sides.

### Dependency direction

```
main.py ──→ graph.py ──→ nodes.py ──→ tools.py ──→ sandbox.py
   │            │           │  │  │         │
   │            │           │  │  │         └──→ config.py
   │            │           │  │  └──→ memory.py ──→ schemas.py
   │            └───────────┘  ├──→ session.py
   └──→ hitl.py ←──────────────┘
   └──→ config.py                    ingest.py ──→ config.py
                                fetch_papers.py ──→ (nothing local)
```

Two deliberate boundaries, both about testability:

- **`sandbox.py` and `hitl.py` import nothing from this project.** That is why they carry the most
  exhaustive tests (7 and 48) — they need no database, no key and no graph to exercise.
- **`config.py` is lazy.** No module-level `model = ChatOpenAI(...)`. Callers invoke `get_model()` at use
  time, so `import nodes` costs nothing and `python main.py --help` answers in 0.18s.
  `tests/test_startup.py` enforces this in a subprocess with no key and a dead DSN. **Never reintroduce a
  module-level `model = get_model()`** — it re-couples the entire codebase to live infrastructure.

### The two persistence layers

Both are Postgres, both provisioned by `init_db()`, both backed by the one `get_pool()` connection pool.

| Layer | Accessor | Keyed by | Holds |
|---|---|---|---|
| Checkpointer | `get_checkpointer()` → `PostgresSaver` | `thread_id` | Per-thread graph state; what makes resume work |
| Store | `get_store()` → `PostgresStore` | namespace tuple | `("thesis_workspace","active_project")` → the proposal + `active_thread_id`; `("thesis_memory", user_id, category)` → `DurableMemoryItem`s |

### The evidence path

Attribution is a contract across four files, pinned by `tests/test_citations.py`:

```
ingest.py            attaches metadata["source"] (filename) + PyPDFLoader's metadata["page"] (0-indexed)
   ↓
tools.format_citation()   renders "filename, p.N"  (1-indexed, hence the +1)
   ↓
search_thesis_literature  prefixes every result with "[i] SOURCE: filename, p.N"
   ↓
research_node prompt      "carry it verbatim into your notes as [filename, p.N]"
   ↓
write_node prompt         "cite every factual claim inline as [source, p.N]" + References section
   ↓
reviewer_node prompt      receives research_notes; judges the draft against them, not its own knowledge
```

---

## 3. Usage

### Setup

```bash
pip install -r requirements.txt
```

Create `.env` in the repo root:

```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o                                                    # optional
JUDGE_MODEL=gpt-4o-mini                                                # optional; must differ from OPENAI_MODEL
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/thesis_db   # optional
```

Postgres must be running with the `pgvector` extension available. Tables and the extension are created on
first run. If the database is unreachable you get an actionable message from
`config.startup_error_message()`, not a stack trace.

### Entry points

```bash
python main.py                # start, or resume the last thread (including at a pending approval gate)
python main.py --new-thread   # fresh conversation, SAME thesis  (--new is an alias)
python main.py --new-project  # different thesis: archives the current proposal first
python fetch_papers.py        # download the 6 arXiv papers in papers/manifest.json
python fetch_papers.py --list # show the corpus without downloading
python ingest.py              # chunk + embed papers/ into pgvector (idempotent)
python ingest.py --rebuild    # drop the collection first (needed after deleting a PDF)
```

A first run, end to end:

```bash
python fetch_papers.py && python ingest.py
python main.py
# Enter your query: the effect of retrieval augmentation on factual accuracy in LLMs
# ... proposal → research → draft → review → approval gate
# Your decision: revise: expand the methodology section
#   -> sending it back for a writing revision
# Your decision: remember: always cite the original paper, never a survey
#   -> saving a durable constraint, then revising
# Your decision: approve
#   -> approving the draft and finalising this thesis
```

### The functions you would actually call

**`config`** — every shared client, constructed once, lazily:

```python
from config import get_model, get_vector_store, init_db

init_db()                                    # once, at startup
model = get_model()                          # cached: same object every call
docs = get_vector_store().similarity_search("self-attention", k=4)
```

**`memory`** — durable constraints that steer every future generation:

```python
from langgraph.store.memory import InMemoryStore
from memory import save_durable_memory, get_durable_memories, count_durable_memories
from schemas import DurableMemoryItem

store = InMemoryStore()
save_durable_memory(store, DurableMemoryItem(
    category="methodology_decision",
    title="Enforce ModernBERT",
    content="Reject Word2Vec baselines.",
    status="enforced",
))
print(get_durable_memories(store))
# • [METHODOLOGY_DECISION - ENFORCED] Enforce ModernBERT: Reject Word2Vec baselines.
print(count_durable_memories(store))   # 1
```

**`hitl`** — the approval-gate command language:

```python
from hitl import parse_human_decision

parse_human_decision("approve.")               # HumanDecision(action='approve', payload='')
parse_human_decision("yes, but fix the intro") # HumanDecision(action='revise', payload='yes, but fix the intro')
parse_human_decision("research: 2024 data")    # HumanDecision(action='research', payload='2024 data')
parse_human_decision("   ")                    # HumanDecision(action='empty', payload='')
parse_human_decision("revise: x").describe()   # 'sending it back for a writing revision'
```

**`session`** — thread and project lifecycle:

```python
from langgraph.store.memory import InMemoryStore
from session import resolve_thread_id, archive_active_proposal

store = InMemoryStore()
thread_id, was_resumed = resolve_thread_id(store)              # (uuid, False) first time
thread_id, was_resumed = resolve_thread_id(store)              # (same uuid, True)
thread_id, _ = resolve_thread_id(store, start_new=True)        # new uuid, saved as active

archive_active_proposal(store)   # -> 'archived_proposal_20260907T120000Z', or None
```

**`sandbox`** — no project dependencies, so it runs anywhere:

```python
from sandbox import run_sandboxed

run_sandboxed("print(6 * 7)")                    # '42'
run_sandboxed("x = 1")                           # 'Execution successful (no output).'
run_sandboxed("raise ValueError('boom')")        # 'Execution Error:\n...boom...'
run_sandboxed("while True: pass", timeout=2)     # 'Execution Error: timed out after 2s.'
```

**`tools`** — citation formatting and the tools themselves (LangChain tools need `.invoke()`):

```python
from tools import format_citation, search_thesis_literature

format_citation({"source": "attention.pdf", "page": 3})   # 'attention.pdf, p.4'   (page is 0-indexed)
format_citation({"source": "notes.md"})                   # 'notes.md'
search_thesis_literature.invoke({"query": "self-attention", "k": 4})
```

**`graph`** — build it yourself if you are embedding rather than using the CLI:

```python
from config import init_db
from graph import build_graph

init_db()                       # migrations are NOT part of build_graph()
graph = build_graph()
result = graph.invoke(
    {"user_query": "...", "messages": [], "thesis_initialized": False,
     "research_notes": [], "review_feedback": [], "revision_count": 0},
    config={"configurable": {"thread_id": "my-thread"}},
)
```

---

## 4. Testing

### Layout

331 tests in 14 files. **299 run by default** — no database, no API calls, no billing. The other 32 are
opt-in because they need a live index and cost money.

| File | Tests | Covers |
|---|---:|---|
| `tests/test_hitl.py` | 48 | The approval-gate command language: every affirmative, qualified affirmatives that must *not* approve, prefix commands, empty input, and the node's dispatch on each. |
| `tests/test_golden_set.py` | 41 | The RAG ground truth: every golden attributes to a real manifest paper, and its quote is re-extracted from the actual PDF page. Skips if `papers/` is unfetched. |
| `tests/test_nodes.py` | 22 | Every routing decision: both `supervisor_router` branches, `research_router`, `proposal_node`'s three paths, `research_node`'s note rule, `write_node`'s prompt, all `reviewer_node` verdicts including the `>= 3` force-approve, all `human_approval_node` commands. |
| `tests/test_silent_failures.py` | 17 | The four bugs that never raised: memory truncation, `--new` semantics, resume-at-gate, ingest duplication. |
| `tests/test_startup.py` | 13 | That importing `graph`/`ingest`/`main` and running `--help` need neither a database nor an API key, plus accessor laziness and `startup_error_message`. |
| `tests/test_citations.py` | 12 | The evidence path hop by hop: `format_citation`, search output, and the three prompts that carry markers through. |
| `tests/test_python_executor.py` | 7 | `run_sandboxed`: stdout, no-output, errors, env scrubbing, timeout, truncation, stdin. |
| `tests/test_session.py` | 3 | `resolve_thread_id` mint/resume/force-new. |
| `tests/test_management.py` | 26 | Memory list/forget, project archive/restore, and the proposal gate. |
| `tests/test_export.py` | 26 | Filename rules, non-destructive writes, and the node paths that export. |
| `tests/test_providers.py` | 24 | Model→provider routing across OpenAI, Anthropic and DeepSeek. |
| `tests/test_render.py` | 21 | LaTeX escaping and structure; the contract when docx/pdf tooling is absent. |
| `tests/test_bibliography.py` | 21 | Citation extraction, and the restraint that stops it inventing entries. |
| `tests/test_context_and_usage.py` | 18 | Trimming without orphaning tool results; token accounting. |
| `tests/test_rag_eval.py` | 16 + 16 | **Opt-in.** The `deepeval` retrieval arm and the `baseline` no-retrieval arm. |

### Commands

```bash
pytest tests/ -v                       # the full default suite (163, ~30s, free)
pytest                                 # same - testpaths = ["tests"] in pyproject.toml

pytest tests/test_hitl.py              # one file
pytest tests/test_hitl.py::test_affirmatives_approve            # one test
pytest "tests/test_hitl.py::test_affirmatives_approve[approve.]"  # one parametrised case
pytest -k "citation and not golden"    # by name
pytest tests/ -q --durations=10        # find the slow ones

pytest -m deepeval                     # RAG eval    (needs Postgres + billed calls)
pytest -m baseline                     # no-retrieval comparison arm
pytest -m ""                           # everything, including both opt-in arms
```

`pyproject.toml` sets `pythonpath = ["."]` — that is what lets bare `pytest` resolve the top-level modules.
Without it every test module fails to import, so don't remove it.

### Writing a new test

There is **no `conftest.py`** and no custom fixtures. Everything uses stock pytest plus four patterns:

**1. Real in-memory stores, not mocks.** `InMemoryStore` implements the same `BaseStore` interface as
`PostgresStore`, so pass it directly and assert on what actually landed:

```python
from langgraph.store.memory import InMemoryStore

def test_something_persists() -> None:
    store = InMemoryStore()
    nodes.proposal_node({"user_query": "idea"}, store)
    assert store.get(session.PROPOSAL_NAMESPACE, session.PROPOSAL_KEY) is not None
```

**2. Swap the accessor, not the client.** Because `config` is lazy, `nodes.py` holds `get_model` as a
module global. Patch it with a **callable** returning your stub:

```python
from unittest import mock

def _model(structured_result=None, invoke_result=None) -> mock.MagicMock:
    m = mock.MagicMock(name="model")
    m.with_structured_output.return_value.invoke.return_value = structured_result
    m.invoke.return_value = invoke_result
    return m

def test_reviewer_approves() -> None:
    stub = _model(structured_result=ReviewDecision(next_step="approved", issues=[], suggestions=[]))
    with mock.patch.object(nodes, "get_model", lambda: stub):     # lambda, not the stub itself
        command = nodes.reviewer_node({"draft": "d", "revision_count": 0})
    assert command.goto == "human_approval_node"
```

Patch targets: `nodes.get_model`, `nodes.get_model_with_tools`, `nodes.interrupt`, `nodes.write_draft`,
`tools.get_vector_store`, `ingest.get_vector_store`, `export.DEFAULT_OUTPUT_DIR`, `render.pdf_toolchain`. Copy the local `_model` helper — it is duplicated across test files on purpose,
so each file reads standalone.

**3. Assert on prompts with a sentinel.** To check something reached the model, put a unique string in the
state and look for it in the captured call:

```python
def test_notes_reach_the_reviewer() -> None:
    stub = _model(structured_result=ReviewDecision(next_step="approved", issues=[], suggestions=[]))
    with mock.patch.object(nodes, "get_model", lambda: stub):
        nodes.reviewer_node({"draft": "d", "research_notes": ["EVIDENCE-SENTINEL"]})
    prompt = stub.with_structured_output.return_value.invoke.call_args[0][0]
    assert "EVIDENCE-SENTINEL" in prompt
```

`write_node` builds a message list, so its prompt is at `stub.invoke.call_args[0][0][0].content`;
`reviewer_node` passes a bare string, so its is `...call_args[0][0]`.

**4. `parametrize` for tables, `monkeypatch` + `tmp_path` for filesystem work.** See
`test_format_citation` and `test_ingestion_passes_deterministic_ids_to_the_vector_store`.

#### Conventions

- **Names are sentences about behaviour**, not method names:
  `test_memories_beyond_the_stores_default_page_are_still_injected`, not `test_get_durable_memories_2`.
- **Comments say why the test exists**, especially when it guards a past bug. Every non-obvious test here
  carries one; keep that up.
- **Type-hint tests**: `def test_x() -> None:`.
- **Group with `# --- section ---`** comment rules inside a file.
- **Mark anything needing a DB or billed calls** with `@pytest.mark.deepeval` or `@pytest.mark.baseline`, and
  import `config` *inside* the test — pytest imports modules during collection to read markers, so a
  module-level import would make even a deselected run connect to Postgres.

#### Before you claim it works

Write the test, watch it pass, then **break the production code and confirm it fails.** Copy the repo to a
temp directory, revert the behaviour, run the file:

```bash
cp *.py pyproject.toml /tmp/mut/ && cp tests/*.py tests/*.json /tmp/mut/tests/
# edit /tmp/mut/<module>.py to undo the fix
cd /tmp/mut && pytest tests/test_thing.py -q
```

This is not ceremony. Every mutation recorded in `CHANGELOG.md` was run, and one of them —
lowering `reviewer_node`'s revision cap from `>= 3` to `>= 2` — passed a green 21-test suite untouched. The
boundary case that catches it exists only because the mutation was tried.
