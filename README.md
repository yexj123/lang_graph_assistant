# Autonomous Multi-Agent Research Assistant

Stateful multi-agent research & synthesis engine powered by LangGraph, featuring dynamic tool orchestration (web search, document parsing, code execution), Pydantic structured routing, and an autonomous reviewer reflection loop.

> **Status:** 🚧 Work in Progress / To be completed
>
> **Development:** From the commit following `2e0821b` onward, this project is developed with
> [Claude Code](https://claude.com/claude-code).

## What it does

A LangGraph-driven CLI that helps draft an academic thesis end to end: it proposes a topic, research
question, and outline from a one-line idea; researches each outline point using a RAG index over your own
PDFs plus live web search and code execution; drafts the writing; runs it through an automated reviewer; and
gates the final draft behind a human approval step. Decisions and constraints you approve are remembered
across sessions.

## Features

- **Multi-agent graph** (proposal → research → write → review → human approval) with conditional routing and
  a bounded revision loop, built on LangGraph.
- **RAG over your literature** — ingest PDFs into a pgvector store; the research agent searches it before
  falling back to the web. Retrieved passages carry a `filename, p.N` marker that survives into the research
  notes and the draft, so every claim can be traced back to a page.
- **Tool use** — web search, local file reading, and Python code generation/execution. The executor runs the
  generated code in a separate process with an allowlisted environment (no API key or database DSN is
  inherited), stdin closed, a 15-second timeout and truncated output. See the security note below for what
  that does *not* cover.
- **Long-term memory** — durable decisions, constraints, and research conclusions persist across sessions in
  Postgres and are injected into every future prompt.
- **Human-in-the-loop** — *two* gates. The **plan** is approved before any research is billed, and the
  **draft** before the graph finishes. Constraints can be listed and deleted from the gate.
- **Any of three providers** — OpenAI, Anthropic or DeepSeek, chosen per model id. Set
  `LLM_MODEL=claude-sonnet-5` and the provider is inferred; no code changes.
- **Live progress and token accounting** — each node prints as it completes, and every turn reports
  calls and tokens (cost too, if you supply prices).
- **Draft export with a bibliography** — approving writes the thesis to
  `drafts/<topic-slug>-<timestamp>.<ext>` in Markdown, LaTeX, DOCX or PDF, with a References section
  built from the draft's own `[source, p.N]` markers and resolved against `papers/manifest.json`.
  `export` at the gate snapshots a version at any point; timestamped, so nothing is overwritten.
- **Session resumability** — the CLI picks up your last conversation automatically, and resumes it *at the
  approval gate* if that is where it stopped.

### Threads vs projects

The thesis proposal outlives conversation threads, so a new thread continues the same thesis. Use
`--new-thread` for a fresh conversation about your current thesis, and `--new-project` to start a different
one — that archives the existing proposal under a timestamped key rather than deleting it.

## Models

Pick a chat model with `LLM_MODEL`; the provider is inferred from the id. `LLM_PROVIDER`
overrides that inference for a model the registry has not seen yet.

| Provider | Set | Example models | Extra install |
|---|---|---|---|
| OpenAI | `OPENAI_API_KEY` | `gpt-4o`, `gpt-4o-mini`, `gpt-4.1` | none |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-opus-5`, `claude-sonnet-5`, `claude-haiku-4-5-20251001` | `pip install langchain-anthropic` |
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek-chat`, `deepseek-reasoner` | none — OpenAI-compatible API |

```bash
LLM_MODEL=claude-sonnet-5 python main.py     # Claude
LLM_MODEL=deepseek-chat   python main.py     # DeepSeek
LLM_PROVIDER=anthropic    python main.py     # that provider's default model
python main.py --status                      # what is actually configured
```

Embeddings stay on OpenAI regardless: the pgvector collection is built with a specific
embedding model and its dimensionality, so switching chat providers must not silently
invalidate the index.

The evaluation judge is separate (`JUDGE_MODEL`) and the eval **fails** if it equals the
generator. A *different provider* is the strongest form of judge independence available —
`JUDGE_MODEL=claude-sonnet-5` while drafting with `gpt-4o`, say.

Optionally set `<PROVIDER>_INPUT_PRICE` / `<PROVIDER>_OUTPUT_PRICE` (USD per million
tokens) to turn the per-turn token count into a cost estimate. Prices are not hardcoded
because they drift, and a stale number is worse than none.

## Security note

**`python_executor` is process isolation, not a security sandbox.** The environment scrubbing is real — the
child process gets an allowlisted environment, so `OPENAI_API_KEY` and `DATABASE_URL` are not reachable from
generated code, and `tests/test_python_executor.py` asserts it. But the child still has **full filesystem and
network access**, so code it runs can read `.env` directly off disk, reach your database, or make outbound
requests. `file_reader` likewise accepts any path the process can read.

Practically: this is a single-operator tool that runs code an LLM wrote from *your* prompts, on your own
machine. That is the threat model it is built for. Do not point it at untrusted input, do not run it as a
service, and do not treat the subprocess as a boundary that contains hostile code — it isn't one. Real
isolation would mean a container, `nsjail`/`firejail`, or a network namespace with resource limits.

## Requirements

- Python 3.11
- PostgreSQL with the [`pgvector`](https://github.com/pgvector/pgvector) extension installed on the server
  (the app creates the extension and its own tables in your database automatically on first run)
- An OpenAI API key

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Create a `.env` file in the repo root:
   ```
   OPENAI_API_KEY=sk-...
   LLM_MODEL=gpt-4o             # optional; also accepts claude-*/deepseek-* (see Models below)
   DATABASE_URL=postgresql://postgres:postgres@localhost:5432/thesis_db   # optional, this is the default
   ```
3. Make sure the Postgres instance in `DATABASE_URL` is reachable. Checkpointing, long-term memory, and
   vector store tables are created automatically the first time you run the app.
4. Build the literature index. `papers/` ships empty, so the research agent has nothing to search until you
   populate it — either drop in your own PDFs, or fetch the reference corpus this repo's evaluation is
   written against:
   ```bash
   python fetch_papers.py   # downloads the 6 arXiv papers in papers/manifest.json, checksum-verified
   python ingest.py         # chunk + embed them into pgvector
   ```

## Usage

```bash
python main.py                # start, or resume the last thread (including at a pending approval gate)
python main.py --new-thread   # fresh conversation on the SAME thesis (--new is an alias)
python main.py --new-project  # a different thesis: archives the current proposal, proposes a new one
python fetch_papers.py        # download the reference corpus named in papers/manifest.json
python ingest.py              # embed PDFs from ./papers/ into the vector store (idempotent)
python ingest.py --rebuild    # drop the collection first, after removing or renaming a PDF
pytest tests/ -v              # fast unit suite - no database, no API calls, no billing
pytest -m deepeval            # RAG quality eval - needs Postgres, an OpenAI key and an ingested corpus
pytest -m baseline            # the no-retrieval comparison arm

python main.py --status               # active thesis, thread, constraints, models
python main.py --list-projects        # archived theses
python main.py --restore-project KEY  # make an archived thesis active again
python main.py --format docx          # export approved drafts as .docx (also tex, pdf)
```

## Evaluation

The RAG pipeline is scored with deepeval against a golden set of 16 questions in
[`tests/golden_set.json`](./tests/golden_set.json). The design decisions that make the numbers worth
anything:

| Concern | How it's handled |
|---|---|
| **Ground truth you can audit** | Every reference answer records the paper, page, and the verbatim passage it came from. `tests/test_golden_set.py` re-extracts that page from the PDF and asserts the quote is really there — so a low score means a bad pipeline, never a bad golden. |
| **Judge independence** | The judge is `JUDGE_MODEL`, and the eval **fails loudly** if it equals `OPENAI_MODEL`. An LLM grading its own output has a documented self-preference bias that inflates faithfulness and relevancy. |
| **Declared thresholds** | Set explicitly in `tests/test_rag_eval.py` (faithfulness 0.8, relevancy 0.7, contextual precision/recall 0.6, correctness 0.7) rather than inheriting deepeval's implicit 0.5. |
| **Attribution** | A no-retrieval baseline arm answers the same questions with no context, so a good score can be credited to retrieval instead of to what the model already knew. |
| **A reproducible corpus** | `papers/manifest.json` pins six arXiv papers by ID and sha256; `fetch_papers.py` reproduces the exact corpus. The PDFs are not committed because arXiv licences are per-paper and redistribution is the author's call. |

To produce the numbers:

```bash
python fetch_papers.py && python ingest.py
JUDGE_MODEL=gpt-4o-mini pytest -m deepeval -v    # retrieval arm
JUDGE_MODEL=gpt-4o-mini pytest -m baseline -v    # no-retrieval comparison
```

> **No scores are published here yet.** Both arms make billed API calls against a live index, so the table
> is left to whoever runs it rather than filled in with numbers nobody can trace to a run.

At the `Enter your query:` prompt, describe your thesis idea, a research request, or a writing request. When
a draft is ready for review, you'll be dropped into an approval gate:

| Input | Effect |
|---|---|
| `approve` | Accept the draft, finalize, and end the session |
| `research: <notes>` | Send it back for more research |
| `revise: <notes>` | Send it back for a writing revision |
| `remember: <rule>` | Save a durable decision/constraint for all future sessions |
| `export` / `save` | Write the current draft to `drafts/` and ask again (`export: <dir>` for a custom path) |
| `memories` | List every durable constraint, numbered, and ask again |
| `forget: <n>` | Delete constraint `<n>` from that listing |
| *(empty)* | Re-asks. Nothing is spent on a rewrite instructed by nothing |
| anything else | Treated as revision feedback |

Approval accepts the obvious variants — `approve`, `Approved!`, `ok.`, `yes`, `lgtm`, `looks good` — but
matches the **whole** input, so `yes, but rework the intro` is feedback rather than an accidental
finalisation. The prefix commands ignore case and spacing around the colon (`Revise : ...` works). Whatever
you type, the CLI prints how it read it before acting, so an interpretation is never silent.

## Project internals

See [`CLAUDE.md`](./CLAUDE.md) for the full architecture: graph shape, the two persistence layers, the
structured-output routing pattern, and known rough edges.

[`DEVELOPER_GUIDE.md`](./DEVELOPER_GUIDE.md) is the orientation for changing the code: what each module is
responsible for, how control and data flow between them, the public functions with worked examples, and how
the test suite is organised and extended.

[`CHANGELOG.md`](./CHANGELOG.md) records the defects found in a hiring-bar review of this repo, how each was
fixed, and how each fix was verified — including a **Known limitations** section listing what was
deliberately left alone.
