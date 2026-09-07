# Changelog

## Unreleased — review remediation

Everything below addresses findings from a hiring-bar code review of this repository at commit `2e0821b`.
Each entry states the defect, the fix, and how the fix was verified. Verification means one of two things:
a measured before/after, or a **mutation check** — the fix was reverted in a throwaway copy of the repo and
the suite was confirmed to fail. A test that passes but cannot fail proves nothing, and several tests in this
list were strengthened only after a mutation slipped past them.

Test suite: **163 passing**, no database, no API calls, no billing. 32 further tests are opt-in
(`-m deepeval`, `-m baseline`) because they need a live index and make billed calls.

---

### 1. The refactored project was never committed

**Defect.** `origin/main` was a broken skeleton. `requirements.txt`, `sandbox.py`, `session.py`, `ingest.py`,
the entire `tests/` directory and `CLAUDE.md` were untracked; `config.py`, `main.py` and `tools.py` were
published in their pre-refactor form, so the public `python_executor` still ran `exec()` in-process with
`OPENAI_API_KEY` in scope — and the hardening that fixed that was one of the untracked files. Seven
`__pycache__/*.pyc` files *were* tracked, and there was no `.gitignore`. Hygiene exactly inverted.

**Fix.** Added `.gitignore` (bytecode, pytest/deepeval caches, `.env`, machine-local Claude settings,
`papers/*.pdf`), untracked the bytecode with `git rm -r --cached`, and staged the full working tree. Added
`papers/.gitkeep`, since git cannot track an empty directory and the ingestion target would otherwise not
exist in a clone.

**Verified.** `git ls-files --cached | grep -c pyc` → 0. Every `*.py` and `requirements.txt` in the working
tree machine-compared against `git ls-files --cached` → 0 missing of 15. `git check-ignore -v` confirmed each
ignore rule fires against a real path.

---

### 2. The documented test command ran zero tests, and the core was untested

**Defect.** `pytest tests/ -v` — the command in the README — failed with `ModuleNotFoundError` on all three
test modules and ran nothing. The repo is a flat set of top-level modules rather than an installed package,
so bare `pytest` put only `tests/` on `sys.path`; `python -m pytest` worked by accident because it adds the
CWD. Anyone following the README concluded the suite was broken. Anyone who fixed it found that `nodes.py` —
249 lines containing every routing decision in the system — had no tests at all, including the
`revision_count >= 3` force-approve that is the graph's only termination guarantee.

**Fix.** `pyproject.toml` sets `pythonpath = ["."]` and `testpaths = ["tests"]`. `test_rag_eval.py` is marked
`deepeval` and deselected by default, and its `from config import ...` moved *inside* the test functions —
load-bearing, because pytest imports a module to discover its markers, so a module-level import would make
even a deselected run pay `config.py`'s import-time database connect. Added `tests/test_nodes.py`: 22 tests
covering both `supervisor_router` branches, all `research_router` cases, `proposal_node`'s
noop/reload/generate paths, `research_node`'s note-recording rule, `write_node`'s prompt assembly, all four
`reviewer_node` verdicts, and all five `human_approval_node` commands.

**Verified.** Before: 3 collection errors, 0 tests. After: 32 passed, 2 deselected, 2.9s, no database.
Mutation checks: lowering the revision cap `>= 3` → `>= 2` and raising it to `>= 4` each fail a different
test, pinning the threshold from both sides; inverting `research_router` fails its test. The first of those
was caught **only** by a boundary case added after the suite first went green.

---

### 3. No error handling in the orchestration layer; import-time database coupling

**Defect.** Zero `try:` blocks in `nodes.py`, `main.py`, `graph.py`, `memory.py` or `config.py`. A single
rate limit ended the session and discarded the turn. Separately, `config.py` constructed every client at
module scope — `ChatOpenAI` raises without `OPENAI_API_KEY`, `PGVector` opens a Postgres connection in its
constructor — and `graph.py` ran `graph = build_graph()` at import, executing migrations. So `import config`
was a blocking side effect, nothing could be imported or unit-tested without live infrastructure, and
`python main.py --help` connected to the database and died on a raw SQLAlchemy traceback after ~23 seconds.

**Fix.** The six shared clients moved behind `@lru_cache` accessors (`get_model`, `get_embeddings`,
`get_pool`, `get_vector_store`, `get_checkpointer`, `get_store`); callers invoke them at use time.
`tools.model_with_tools` became `get_model_with_tools()`. `build_graph()` no longer runs `init_db()` —
`main.py` calls migration and construction explicitly, in order. `main.py` keeps only `argparse` at module
scope. Startup failures raise `SystemExit` with `config.startup_error_message()`, which names the unreachable
DSN or the missing key and always appends the original exception. Each CLI turn is wrapped so a transient
failure reports and returns to the prompt; the blind `state_snapshot.tasks[0].interrupts[0]` index is
guarded; Ctrl-C and EOF exit cleanly.

**Verified.** `python main.py --help`: 22.7s crash → **0.18s**, prints help. `import main`: required DB and
key → 0.07s, neither. Startup with the DB down now reports `Cannot reach Postgres at <dsn>` plus
`PoolTimeout: couldn't get a connection after 30.00 sec` instead of a stack trace. Added
`tests/test_startup.py`, which imports `graph`, `ingest` and `main` in a subprocess with no
`OPENAI_API_KEY` and a `DATABASE_URL` pointing at a closed port. Mutation check: restoring the eager
singletons fails 11 tests. The `sys.modules` stub that `test_nodes.py` had needed was deleted — it existed
only to work around this coupling.

---

### 4. The RAG path could not be demonstrated or trusted

**Defect.** Four separate problems in the feature the README leads with.

- `reviewer_node`'s prompt contained topic, question, outline and draft — **no source material** — while its
  schema asked it to return `needs_more_research` for factual gaps. Every factual verdict it produced came
  from parametric memory. The "autonomous reviewer reflection loop" was checking prose against an outline.
- `ingest.py` had always attached page metadata with the comment *"for agent citation"*, and
  `search_thesis_literature` had always discarded it. No prompt anywhere mentioned citations. An academic
  writing tool could not produce a traceable reference — its core requirement.
- `papers/` shipped empty with no manifest, so nobody but the author could exercise retrieval.
- The evaluation was two hand-written Q&A pairs whose reference answers cited nothing, about papers not in
  the repo and named nowhere, graded by **the same `gpt-4o` that generated the answers**, against deepeval's
  implicit 0.5 thresholds, with no baseline. A low score could not be distinguished from a bad golden.

**Fix.** `reviewer_node` now receives `research_notes` and is told to judge against that material rather than
its own knowledge. `tools.format_citation()` renders source plus PyPDFLoader's 0-indexed `page` as
`filename, p.N`; the research prompt requires that marker to be carried verbatim into the notes; the write
prompt requires inline `[source, p.N]` citations and a References section, and forbids inventing them.
`papers/manifest.json` pins six arXiv papers by id, filename and sha256, and `fetch_papers.py` downloads and
checksum-verifies them — idempotent, rate-paced, self-identifying; the PDFs stay gitignored because arXiv
licences are per-paper and redistribution is the author's call. The golden set moved to
`tests/golden_set.json`, expanded to 16 questions, each recording the paper, page and verbatim passage its
reference answer came from. The judge is `JUDGE_MODEL` and the run **fails loudly** if it equals
`OPENAI_MODEL`. Thresholds are declared explicitly. A `-m baseline` arm answers the same questions with no
retrieval, so a good score can be attributed to retrieval rather than to what the model already knew.

**Verified.** `tests/test_golden_set.py` re-extracts each cited page from the real PDF and asserts the quote
is actually there — 41 tests, all passing, which is what makes the ground truth auditable rather than merely
claimed. Every arXiv id was checked against the arXiv API for a resolving 200 and its real title before
entering the manifest; the corpus was downloaded (6 files, ~11MB, all valid `%PDF`) and re-running verified
checksums without re-downloading. `tests/test_citations.py` pins each hop of the evidence path, including the
`bool`-is-a-subclass-of-`int` case that would otherwise render `page=True` as `p.2`. The judge guard was
verified firing: setting `JUDGE_MODEL=OPENAI_MODEL=gpt-4o` fails with an explicit message.

No evaluation scores are published. Both arms need a live index and billed calls, so the numbers are left to
whoever runs them rather than filled in with figures nobody can trace to a run.

---

### 5. Four silent failures

**Defect.** None of these raised. They quietly made the output wrong, which is the worst failure mode for a
tool whose pitch is that it remembers.

- `get_durable_memories` relied on `BaseStore.search`'s default `limit=10`, so the eleventh constraint in a
  category silently stopped being injected into any prompt. A rule the user explicitly saved stopped being
  enforced, with nothing to indicate it.
- `--new` minted a fresh thread, but `proposal_node` reloads the saved proposal on any thread with no topic
  yet — so a brand-new thesis idea silently produced the old thesis, and there was no way to start a second
  project at all.
- Resuming a thread parked at the approval gate swallowed the first typed query. Invoking LangGraph with new
  input at an interrupt writes the input into state and re-fires the interrupt **without running the node**.
- `ingest.py` appended the whole corpus again under fresh UUIDs on every run, doubling the index and skewing
  top-k retrieval, since duplicate chunks crowd out the results they displace.

**Fix.** `memory.py` pages through with an explicit limit, caps the injected block at
`MAX_INJECTED_MEMORIES`, and **says so in the prompt text** when it truncates — capping silently was the bug.
`CATEGORIES` is derived from the `DecisionCategory` Literal via `get_args` rather than retyped. `--new` split
into `--new-thread` (fresh conversation, same thesis; `--new` kept as an alias) and `--new-project`, which
calls `session.archive_active_proposal()` to move `current_proposal` to a timestamped `archived_proposal_*`
key — archived, never deleted. `PROPOSAL_NAMESPACE`/`PROPOSAL_KEY` moved to `session.py` and are imported by
`nodes.py`, so the archive cannot drift onto a key the graph does not read. `run_cli` checks
`graph.get_state(config).next` before prompting and drives the gate through `_service_interrupts`. Chunk ids
are now `sha256(source, page, text)` and PGVector upserts on conflict, making ingestion idempotent; source
and page are in the hash because identical boilerplate on different pages would otherwise collapse and drop
real chunks. `ingest.py --rebuild` drops the collection first, the only way to evict chunks whose PDF is gone.

**Verified.** `tests/test_silent_failures.py`, 17 tests. The LangGraph interrupt behaviour is characterised
directly, not just fixed, so the next person to touch `run_cli` can see why the `.next` check exists.
Mutation checks: restoring the implicit `limit=10` fails 3 tests; non-deterministic chunk ids fail 2;
removing the `.next` check fails 1; drifting the proposal keys apart fails 1.

---

### 6. `python_executor` was described as sandboxed

**Defect.** The README called the code executor "sandboxed", which an outside reader would take as a security
boundary. The environment scrubbing is real, but the child process has full filesystem and network access —
it can read `.env` directly off disk and make outbound requests. `file_reader` likewise accepts any readable
path. Describing that as a sandbox is the kind of claim that does not survive an interview.

**Fix.** The feature bullet now states precisely what the executor does (separate process, allowlisted
environment, stdin closed, 15s timeout, truncated output) and points to a new **Security note** that states
plainly what it does not cover, names the actual threat model — a single-operator tool running LLM-written
code from the operator's own prompts — and says what real isolation would require.

**Verified.** Documentation change; the behaviour was already covered by `tests/test_python_executor.py`,
which asserts the environment scrubbing it does provide.

---

### 7. Line endings depended on each contributor's git config

**Defect.** No `.gitattributes`. Whether a file landed in the repository with CRLF depended on the
`core.autocrlf` setting of the machine that committed it. This repo is authored on Windows and would be
cloned on Linux — an invitation to spurious whole-file diffs.

**Fix.** Added `.gitattributes`: `* text=auto`, `*.pdf binary` (so a stray `git add -f` cannot corrupt a
checksum-verified download), and `eol=lf` pinned for `*.json` and `*.sh`.

**Verified.** `git ls-files --eol` reports `i/lf` for all 30 indexed files — the stored content was already
correct, so this changes no bytes; it makes that outcome independent of local configuration. Note that
`git add` still prints *"LF will be replaced by CRLF"* on Windows: that is git announcing the intended
working-tree conversion, not a problem.

---

### 8. The approval gate misread its own command language

**Defect.** The human-in-the-loop parser was an order-dependent chain of `startswith` checks ending in an
exact-match set `{"approve", "approved", "ok", "yes", "looks good"}`. Typing **`approve.`** missed that set
and fell through to "treat anything unrecognised as revision feedback" — so accepting a draft with a full
stop silently spent a billed LLM call rewriting the document the operator had just approved, and nothing said
so. Empty input became a revision instructed by nothing. The fallthrough was invisible in every case: a
question like *"what did the reviewer object to?"* was fed back as an edit instruction.

**Fix.** New `hitl.py`, importing nothing beyond `re`/`dataclasses`/`typing` — the same reasoning that keeps
`sandbox.py` dependency-free, so the entire command language is unit-testable without a database and
`main.py` can import it at module scope without breaking the `--help` guarantee. `parse_human_decision()`
returns a `HumanDecision(action, payload)`. Prefix commands tolerate case and spacing around the colon.
Approval is matched against the **whole** normalised input with trailing punctuation stripped: `approve.` and
`OK!` approve, while `yes, but rework the intro` does not — approval is the one irreversible branch, so it is
liberal about punctuation and strict about qualification. Empty input returns
`Command(goto="human_approval_node")` and re-asks. `main.py` prints `decision.describe()` before resuming, so
no interpretation is silent.

**Verified.** `tests/test_hitl.py`, 48 tests enumerating the command language. Mutation checks: restoring the
exact-match set fails 18 tests; making approval a prefix match — which would let *"yes, but rework the
introduction"* end the thesis — fails 4; making empty input fall through to a rewrite fails 4. That a node
may re-interrupt itself without deadlocking was confirmed experimentally against `InMemorySaver` before the
fix relied on it.

---

## Known limitations

Deliberately not addressed. Recorded so they are choices rather than oversights.

- **The proposal is never put to the human.** `proposal_node` invents the topic, research question and
  outline, then routes straight into research or writing. The human first sees the plan bundled inside a
  draft review. For a thesis tool the plan is the expensive thing to get wrong, so the gate is arguably on
  the wrong step.
- **`supervisor_router` routes on the raw query string alone** — no system prompt, no thesis state, no
  memory — and there is no "just answer" or "we're done" route, so *"what's my research question again?"*
  spins up the full research → write → review → approve machine and a billed draft.
- **`messages` uses `operator.add`, not LangGraph's `add_messages` reducer**, and neither `messages` nor
  `research_notes` is ever trimmed or summarised. In a system designed to loop, context grows monotonically.
- **`DB_URI.replace("postgresql://", ...)`** mis-handles a `postgres://` scheme or an already-qualified DSN
  instead of failing loudly.
- **`ThesisState.request_type`** is declared and initialised but read by nothing.
- **`langchain-community` is deprecated** (`tools.py` emits a `DeprecationWarning`); `DuckDuckGoSearchRun`
  needs migrating to its standalone package.
- **No evaluation scores are published**, per the reasoning in item 4.
