# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

An autonomous multi-agent thesis research & writing assistant built on LangGraph. A supervisor routes user
queries between a research agent (RAG over ingested PDFs + web search + code execution) and a writing agent,
with a reviewer agent gating drafts before they reach a human-in-the-loop approval step. Status: work in
progress (per README.md) — expect rough edges, not a finished product.

## Running

Dependencies are pinned in `requirements.txt` (`pip install -r requirements.txt`). `pyproject.toml` exists
but holds *only* `[tool.pytest.ini_options]` — there is no packaging metadata and no lint config, so don't
assume `pip install -e .` or a command like `ruff` works. The interpreter actually used for this project is the `llm-finetune`
conda env (Python 3.11) — a shared general-purpose env, not one dedicated to this repo, so `pip list` there
will show unrelated packages too. There's no `.venv`/project-local env and no `environment.yml`.

Requires a Postgres instance (with the `pgvector` extension available) and a `.env` file in the repo root with:
- `OPENAI_API_KEY` (required, read implicitly by `langchain_openai`)
- `OPENAI_MODEL` (optional, defaults to `gpt-4o`)
- `DATABASE_URL` (optional, defaults to `postgresql://postgres:postgres@localhost:5432/thesis_db`)

```bash
python main.py                # interactive CLI - the primary entry point. Auto-resumes the last thread, and
                              # if that thread stopped at the approval gate it services the gate BEFORE
                              # prompting: invoking with fresh input at an interrupt writes the query into
                              # state and re-fires the interrupt without ever acting on it.
python main.py --new-thread   # fresh thread, same thesis (--new is kept as an alias)
python main.py --new-project  # archives the active proposal, then proposes a new thesis
python fetch_papers.py        # download the 6 arXiv papers pinned in papers/manifest.json (id + sha256).
                              # PDFs stay gitignored - the manifest is the reproducibility contract, not the
                              # binaries. --list shows the corpus, --write-checksums is maintainer-only.
python ingest.py              # chunk + embed PDFs into pgvector. Idempotent: chunk ids are sha256 of
                              # (source, page, text) and PGVector upserts on conflict, so re-running does
                              # not duplicate. --rebuild drops the collection first, the only way to evict
                              # chunks whose source PDF is gone. papers/ ships empty,
                              # so search_thesis_literature returns nothing until you fetch or add PDFs.
pytest tests/ -v              # fast unit suite: sandbox, session, and all of nodes.py. No DB, no API calls.
                              # `pythonpath = ["."]` in pyproject.toml is what makes bare `pytest` resolve the
                              # top-level modules; without it every test module fails to import.
pytest -m deepeval            # RAG eval against the 16 sourced goldens in tests/golden_set.json. Judge is
                              # JUDGE_MODEL and the run FAILS if it equals OPENAI_MODEL (self-grading inflates
                              # faithfulness/relevancy). Thresholds are declared in test_rag_eval.py, not
                              # inherited from deepeval's implicit 0.5. Needs a live DB + billed calls.
pytest -m baseline            # no-retrieval arm: same questions with no context, measured not gated, so a
                              # good deepeval score can be attributed to retrieval rather than parametric recall
```

## Architecture

Everything is wired through a single `StateGraph(ThesisState)` (`graph.py`), where `ThesisState` (`state.py`) is
the one TypedDict threaded through every node. `messages` and `research_notes` are append-only
(`Annotated[..., operator.add]`); everything else is overwritten by whichever node returns it.

**Graph shape** — static edges only get you partway; `reviewer_node` and `human_approval_node` do their own
routing via `Command(goto=...)`, so the actual control flow isn't fully visible in `graph.py`'s edge list:

```
START -> proposal_node -> [supervisor_router] -> research_node | write_node
research_node -> [research_router] -> tool_node -> research_node (loop until no tool_calls)
                                    -> write_node
write_node -> reviewer_node -> (Command.goto) -> research_node | write_node | human_approval_node
human_approval_node -> (Command.goto, via interrupt()) -> research_node | write_node | END
```

`reviewer_node` force-approves after 3 revisions (`revision_count`) regardless of quality, to guarantee
termination.

**Two distinct persistence layers**, both Postgres-backed and both provisioned via `init_db()` in `config.py`,
which `main.py` calls once at startup before `build_graph()`. Migration is deliberately *not* part of
`build_graph()`, and there is no module-level `graph = ...`, so importing `graph.py` performs no I/O:
- `checkpointer` (`PostgresSaver`) — per-thread conversation/graph state, keyed by `thread_id`. `main.py`
  auto-resumes the last thread by default (via `session.resolve_thread_id`, see below); `--new-thread` forces
  a fresh one. A resumed thread may be parked mid-graph at `human_approval_node`, so `run_cli` checks
  `graph.get_state(config).next` before prompting and drives the gate via `_service_interrupts` — see the
  command notes above for why invoking with a fresh query there would silently do nothing.
- `store` (`PostgresStore`) — cross-session long-term memory, organized by namespace tuple:
  - `("thesis_workspace", "active_project")` — holds both the single active thesis proposal
    (topic/question/outline, written once by `proposal_node` and reloaded on future runs so the proposal step
    is skipped) and, under key `"active_thread_id"`, the last-used `thread_id` (`session.py`). Still one active
    project at a time, so "resume" means "carry on with the one ongoing thesis". The proposal reload fires
    on any thread that has no topic yet, which is why a new *thread* keeps the same thesis; `--new-project`
    calls `session.archive_active_proposal()`, moving `current_proposal` to a timestamped
    `archived_proposal_*` key (archived, never deleted) so the next run proposes fresh.
    `PROPOSAL_NAMESPACE`/`PROPOSAL_KEY` live in `session.py` and are imported by `nodes.py` — don't
    re-inline those literals, or the archive will move a key the graph doesn't read.
  - `("thesis_memory", user_id, category)` — durable decisions/constraints (`DurableMemoryItem` in
    `schemas.py`), one namespace per `DecisionCategory`. `memory.py` wraps read/write of these,
    paginating explicitly: `BaseStore.search` defaults to `limit=10`, which silently dropped every
    constraint past the tenth in a category, so a saved rule just stopped being enforced. The injected
    block is capped at `MAX_INJECTED_MEMORIES` and says so in the text when it truncates. `CATEGORIES`
    is derived from the `DecisionCategory` Literal via `get_args`, so a category added to the schema
    cannot end up written-but-never-read.
    `get_durable_memories()` is injected into the system prompt of `research_node`, `write_node`, and the
    human approval payload, so anything saved here silently steers all future generations.
  - `user_id` is hardcoded to `"default_user"` everywhere (`memory.py`). Deliberate, not a gap: this is a
    single-operator CLI tool with no auth, so there's nothing to isolate. Don't build multi-user support here
    without a real reason to point more than one person's CLI at the same Postgres instance.

**Structured output over free-text parsing** — every routing/classification/extraction point uses
`model.with_structured_output(<PydanticSchema>)` against a schema in `schemas.py`, rather than parsing LLM
prose: `SupervisorDecision` (routing), `ThesisProposal` (initial proposal), `ReviewDecision` (verdict + issues
+ suggestions), `DurableMemoryItem` (memory extraction), `GeneratedCode` (the `code_generator` tool). Follow
this pattern for any new decision point instead of ad-hoc string matching.

**The evidence path** — retrieval carries attribution end to end, and each hop is pinned by
`tests/test_citations.py`. `ingest.py` attaches `source` (filename) and PyPDFLoader attaches `page`
(0-indexed); `tools.format_citation()` renders those as `filename, p.N` (1-indexed, hence the +1) and
`search_thesis_literature` prefixes every result with it. `research_node`'s prompt tells the agent to copy
that marker verbatim into its notes; `write_node`'s prompt requires inline `[source, p.N]` citations plus a
References section and forbids inventing them; `reviewer_node` receives `research_notes` in its prompt and is
told to judge the draft against that material rather than its own knowledge. That last part matters: the
reviewer's schema asks it to return `needs_more_research` for factual gaps, which it cannot honestly do
without the sources in front of it. Don't remove notes from the reviewer prompt to save tokens.

**Human-in-the-loop protocol** (`human_approval_node` in `nodes.py`, driven from `main.py`'s CLI loop via
`graph.get_state(config).next` + `Command(resume=...)`) — the human's free-text reply is parsed by
`hitl.parse_human_decision()` into a `HumanDecision(action, payload)`. `hitl.py` deliberately imports nothing
beyond `re`/`dataclasses`/`typing`, on the same reasoning as `sandbox.py`: the command language is fully
unit-testable without a DB, and `main.py` can import it at module scope without breaking the "--help touches
no infrastructure" rule.

Resolution order: `remember:`/`research:`/`revise:` prefixes (case- and spacing-insensitive) → approval →
empty → revision feedback. Two rules matter and are pinned by `tests/test_hitl.py`:

- **Approval matches the whole normalised input**, with trailing punctuation stripped. `approve.` and `OK!`
  approve; `yes, but rework the intro` does not. Approval is the one irreversible branch — it ends the graph
  and writes a `research_conclusion` memory — so it must be liberal about punctuation and strict about
  qualification. The old exact-match set `{"approve","approved","ok","yes","looks good"}` got this backwards:
  `approve.` fell through and silently triggered a billed rewrite.
- **Empty input returns `Command(goto="human_approval_node")`**, re-firing the interrupt to re-ask. A node may
  re-interrupt itself; it does not deadlock. Previously empty input became a revision with empty feedback.

`main.py` echoes `decision.describe()` before resuming, so the fallthrough to revision feedback is visible at
the moment of decision. New human-facing commands go in `hitl.py` plus the dispatch in `nodes.py`, not the CLI.

**Tools** (`tools.py`, bound only to the research agent via `get_model_with_tools()`): `search_thesis_literature`
(pgvector RAG — its own docstring tells the model to prefer this over `web_search`), `web_search`
(DuckDuckGo), `file_reader` (txt/md/py/json/pdf), `code_generator` (structured-output Python generation),
`python_executor` (delegates to `sandbox.run_sandboxed` — runs the code in its own `python` subprocess with
an allowlisted environment (no inherited secrets), a 15s timeout, stdin disabled, and truncated output; still
has full filesystem/network access from that child process, so treat any change here as security-sensitive).
`sandbox.py` has no dependency on `config.py`/Postgres/OpenAI by design, specifically so it stays unit-testable
without a live DB — see `tests/test_python_executor.py`.

**Config as DI root** — `config.py` owns every shared client behind an `@lru_cache` accessor: `get_model()`,
`get_embeddings()`, `get_vector_store()`, `get_pool()`, `get_checkpointer()`, `get_store()`. All other modules
call these rather than constructing their own, and the cache is what makes "one pool, one model" true.

Accessors are lazy on purpose, and this is load-bearing rather than stylistic: `ChatOpenAI` raises without
`OPENAI_API_KEY` and `PGVector` opens a connection inside its constructor, so eager module-level singletons
made `import config` require both a live database and a valid key. Call the accessor at use time; never
reintroduce a module-level `model = get_model()`. `tests/test_startup.py` enforces this by importing `graph`,
`ingest` and `main` in a subprocess with no key and a dead `DATABASE_URL`.

## Known rough edges

- `vector_store` in `config.py` is a `langchain_postgres.PGVector` (not `PGVectorStore` — that class exists in
  the installed version but takes a completely different `PGEngine`-based constructor; using it with
  `embeddings=`/`collection_name=`/`connection=` kwargs raises `TypeError`). `get_vector_store()` still blocks
  on first call, because PGVector connects in its constructor to create the `vector` extension — it is deferred,
  not non-blocking. `connect_args={"connect_timeout": 5}` is per connection attempt and SQLAlchemy makes
  several (it tries both the `::1` and `127.0.0.1` resolutions of `localhost`), so the observed wall-clock
  failure is 20-50s, not the ~10s you would predict from the setting. `main.py` and `ingest.py` catch that
  failure and print an actionable message via `config.startup_error_message`.
- `ThesisState.request_type` is declared and initialized (`main.py`) but no node or router currently reads or
  sets it — don't assume it does anything if you see it referenced.
- `research_lang_graph.ipynb`, the pre-refactor notebook this codebase was split out of, is no longer in the
  tree — it was removed after the split, since the `.py` files are the source of truth and git history still
  has it. Recover it with `git show 2e0821b:research_lang_graph.ipynb` if you ever need to compare against
  the original; don't reintroduce it as a second implementation.
