"""Interactive CLI for the thesis assistant.

Every heavy import lives inside run_cli(). Argument parsing must not require a
database or an API key — `python main.py --help` previously imported config and
graph at module scope, connected to Postgres, and died on a raw SQLAlchemy
traceback after ~25 seconds.
"""

import argparse

# Safe at module scope: hitl imports only re/dataclasses/typing, so it costs nothing and
# does not compromise the "--help must not touch infrastructure" guarantee above.
from hitl import parse_human_decision

APPROVAL_HELP = """
Action Commands:
  • 'approve'           -> Accept draft and finalize.
  • 'research: <notes>' -> Request more research.
  • 'revise: <notes>'   -> Request writing revision.
  • 'remember: <rule>'  -> Save constraint/decision to long-term memory.
  • 'export'            -> Write the current draft to drafts/, then ask again.
  • 'memories'          -> List durable constraints, then ask again.
  • 'forget: <n>'       -> Delete constraint <n> from that listing.
  • Any other text      -> Treated as revision feedback."""


def run_cli(new_thread: bool = False, new_project: bool = False, export_format: str = "md") -> None:
    # Deferred so that --help, and any import of this module, stay free of I/O.
    from langchain_core.messages import HumanMessage
    from langgraph.types import Command

    from config import get_pool, get_store, init_db, startup_error_message
    from graph import build_graph
    from session import archive_active_proposal, resolve_thread_id
    from usage import Usage

    import config as _config
    import export as _export

    usage = Usage()
    _export.DEFAULT_FORMAT = export_format

    print("=== Academic Research & Writing Assistant Initialized ===")
    print("Type 'exit' or 'quit' to end the session.\n")

    try:
        init_db()
        graph = build_graph()
        store = get_store()
        pool = get_pool()
    except Exception as exc:  # noqa: BLE001 - any failure here is fatal and must be readable
        raise SystemExit(startup_error_message(exc)) from exc

    with pool:
        if new_project:
            # A new thread alone is not enough: proposal_node reloads the saved proposal
            # on any thread that has no topic yet, so without this the "new" project
            # silently inherits the old thesis.
            archive_key = archive_active_proposal(store)
            if archive_key:
                print(f"Previous thesis proposal archived as '{archive_key}'.")
            print("Starting a new thesis project.\n")

        session_id, was_resumed = resolve_thread_id(
            store, start_new=new_thread or new_project
        )
        config = {"configurable": {"thread_id": session_id}}
        snapshot = graph.get_state(config)
        resumed_state = snapshot.values
        thread_has_state = bool(resumed_state)

        if thread_has_state:
            print(f"Resuming previous session ({session_id}).")
            print(f"  Thesis topic: {resumed_state.get('thesis_topic') or '(not yet proposed)'}")
            print(f"  Last review status: {resumed_state.get('review_status') or '(none yet)'}\n")

            # This thread stopped mid-graph at the approval gate. Invoking with a fresh
            # query here would write it into state and re-fire the interrupt without ever
            # acting on it - the typed query would just vanish. Service the gate first.
            if snapshot.next:
                print("This session stopped at the approval gate. Picking it up there.\n")
                try:
                    _service_interrupts(graph, config, Command, usage)
                except KeyboardInterrupt:
                    print("\n[interrupted] Approval abandoned; the gate is still pending.")
                except Exception as exc:  # noqa: BLE001
                    print(f"\n[error] Could not resume the approval gate: {type(exc).__name__}: {exc}")
        elif was_resumed:
            # A thread id was saved from a prior run, but nothing was ever submitted on it
            # (e.g. the process exited before the first query) - nothing to resume.
            print(f"Session ({session_id}) has no prior activity yet. Starting fresh.\n")
        else:
            print(f"Starting a new session ({session_id}).\n")

        while True:
            try:
                query = input("\nEnter your query: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nSession terminated.")
                break

            if query.lower() in {"exit", "quit"}:
                print("Session terminated.")
                break
            if not query:
                continue

            if not thread_has_state:
                input_state = {
                    "user_query": query,
                    "messages": [HumanMessage(content=query)],
                    "thesis_initialized": False,
                    "thesis_topic": "",
                    "research_question": "",
                    "outline": [],
                    "proposal_approved": False,
                    "proposal_feedback": [],
                    "research_notes": [],
                    "draft": "",
                    "draft_path": "",
                    "request_type": "",
                    "review_status": "",
                    "review_feedback": [],
                    "revision_count": 0,
                }
                thread_has_state = True
            else:
                input_state = {
                    "user_query": query,
                    "messages": [HumanMessage(content=query)],
                    "review_feedback": [],
                }

            try:
                _run_turn(graph, config, input_state, Command, usage)
                print(f"  [usage] {usage.summary(_config.MODEL_PROVIDER.name)}")
            except KeyboardInterrupt:
                print("\n[interrupted] Turn abandoned. The thread is checkpointed, nothing is lost.")
            except Exception as exc:  # noqa: BLE001 - a bad turn must not end the session
                print(f"\n[error] This turn failed: {type(exc).__name__}: {exc}")
                print("The thread is checkpointed, so your work is intact. Try again or rephrase.")


class _NullUsage:
    """Stand-in when a caller has no session usage tracker (e.g. resuming at startup)."""

    def record(self, *_args, **_kwargs) -> None:
        return None


_NODE_LABELS = {
    "proposal_node": "drafting a thesis proposal",
    "proposal_approval_node": "waiting for your approval of the plan",
    "research_node": "researching",
    "tool_node": "running a tool",
    "write_node": "writing the draft",
    "reviewer_node": "reviewing the draft",
    "human_approval_node": "waiting for your approval of the draft",
}


def _stream(graph, payload, config: dict, usage) -> None:
    """Run the graph, printing each node as it completes.

    `invoke` returns only when the whole turn finishes, which for a research loop means
    minutes of silence while billed calls run - indistinguishable from a hang. `stream`
    with stream_mode="updates" yields one item per completed node, which is also where
    token usage can be harvested from the messages each node returns.
    """
    for update in graph.stream(payload, config=config, stream_mode="updates"):
        for node_name, node_state in (update or {}).items():
            label = _NODE_LABELS.get(node_name, node_name)
            print(f"  ... {label}")
            for message in (node_state or {}).get("messages", []) or []:
                usage.record(message)


def _run_turn(graph, config: dict, input_state: dict, Command, usage) -> None:
    """Run one graph invocation, then service any human-approval interrupts it raises."""
    _stream(graph, input_state, config, usage)
    _service_interrupts(graph, config, Command, usage)


def _render_proposal_gate(payload: dict) -> None:
    """The plan gate. Shown before any research is billed."""
    print("\n" + "=" * 60)
    print("                  THESIS PLAN APPROVAL")
    print("=" * 60)
    print(f"Topic            : {payload.get('thesis_topic', '')}")
    print(f"Research question: {payload.get('research_question', '')}")
    print("\nOutline:")
    for index, section in enumerate(payload.get("outline", []), 1):
        print(f"  {index}. {section}")

    constraints = payload.get("active_constraints")
    if constraints and constraints != "No durable decisions recorded yet.":
        print("\nActive Long-Term Constraints:")
        print(constraints)

    print("\nApprove to begin research, or describe what to change about the plan.")
    print("Nothing has been researched or billed yet.")


def _service_interrupts(graph, config: dict, Command, usage=None) -> None:
    """Drive the approval gates until the graph stops asking.

    Split out from _run_turn because a thread can already be sitting at a gate when the
    CLI starts, and that has to be handled before prompting for anything new.
    """
    state_snapshot = graph.get_state(config)
    while state_snapshot.next:
        interrupts = state_snapshot.tasks[0].interrupts if state_snapshot.tasks else ()
        if not interrupts:
            # Paused without an interrupt payload (e.g. a recursion limit). There is
            # nothing to ask the human, so hand control back rather than IndexError.
            print("\n[warn] Graph paused with no approval request pending; returning to the prompt.")
            return

        payload = interrupts[0].value

        # Two different gates share this loop; they ask for different things.
        if isinstance(payload, dict) and payload.get("kind") == "proposal":
            _render_proposal_gate(payload)
            user_action = input("\nYour decision: ").strip()
            print(f"  -> {parse_human_decision(user_action).describe()}")
            _stream(graph, Command(resume=user_action), config, usage or _NullUsage())
            state_snapshot = graph.get_state(config)
            continue

        print("\n" + "=" * 60)
        print("                  HUMAN APPROVAL GATE")
        print("=" * 60)
        print(f"Reviewer Verdict: {payload.get('review_status', 'N/A')}")

        feedback_items = payload.get("review_feedback", [])
        if feedback_items:
            print("\nFeedback:")
            for item in feedback_items:
                print(f"  {item}")

        active_constraints = payload.get("active_constraints")
        if active_constraints and active_constraints != "No durable decisions recorded yet.":
            print("\nActive Long-Term Constraints:")
            print(active_constraints)

        exported = payload.get("draft_path")
        if exported:
            print(f"\nLast exported to: {exported}")

        print("\nCurrent Generated Draft:")
        print("-" * 60)
        print(payload.get("current_draft", "No draft content available."))
        print("-" * 60)
        print(APPROVAL_HELP)

        user_action = input("\nYour decision: ").strip()

        # Say what the input was understood as. Unrecognised text is treated as revision
        # feedback - a reasonable default, but it used to happen silently, including for
        # "approve." which missed the old exact-match set and triggered a full rewrite.
        print(f"  -> {parse_human_decision(user_action).describe()}")

        _stream(graph, Command(resume=user_action), config, usage or _NullUsage())
        state_snapshot = graph.get_state(config)

    # The graph has finished. Say where the thesis ended up - this is the only moment the
    # operator learns the run produced a file at all.
    final_path = graph.get_state(config).values.get("draft_path")
    if final_path:
        print(f"\nDraft written to: {final_path}")


def _with_store(action) -> int:
    """Run a one-shot command that needs the store, with the same startup diagnostics."""
    from config import get_pool, get_store, init_db, startup_error_message

    try:
        init_db()
        store = get_store()
        pool = get_pool()
    except Exception as exc:  # noqa: BLE001
        print(startup_error_message(exc))
        return 1

    with pool:
        return action(store)


def print_status() -> int:
    """Everything about the current project, without entering the interactive loop."""
    import config
    import providers
    from memory import count_durable_memories, format_memory_listing, list_durable_memories
    from session import PROPOSAL_KEY, PROPOSAL_NAMESPACE, list_archived_proposals, resolve_thread_id

    print("Models")
    print(f"  chat  : {config.MODEL_NAME}  (provider: {config.MODEL_PROVIDER.name}, "
          f"key {config.MODEL_PROVIDER.api_key_env})")
    print(f"  judge : {config.JUDGE_MODEL_NAME}  (provider: {config.JUDGE_PROVIDER.name})")
    print(f"  embed : {config.EMBEDDING_MODEL_NAME}  (always OpenAI)")
    print(f"\nAvailable providers:\n{providers.describe_available()}")
    print(f"\nDatabase\n  {config.DB_URI}")

    def report(store) -> int:
        thread_id, resumed = resolve_thread_id(store)
        proposal = store.get(PROPOSAL_NAMESPACE, PROPOSAL_KEY)
        print(f"\nThread\n  {thread_id} ({'resumed' if resumed else 'new'})")

        print("\nActive thesis")
        if proposal:
            print(f"  topic   : {proposal.value.get('thesis_topic', '')}")
            print(f"  question: {proposal.value.get('research_question', '')}")
            for i, section in enumerate(proposal.value.get("outline", []), 1):
                print(f"    {i}. {section}")
        else:
            print("  (none - your next query will propose one)")

        archived = list_archived_proposals(store)
        if archived:
            print(f"\nArchived projects ({len(archived)}) - restore with --restore-project <key>")
            for item in archived:
                print(f"  {item['key']}  {item.get('thesis_topic', '')}")

        print(f"\nDurable constraints ({count_durable_memories(store)})")
        print(format_memory_listing(list_durable_memories(store)))
        return 0

    return _with_store(report)


def print_projects() -> int:
    from session import list_archived_proposals

    def report(store) -> int:
        archived = list_archived_proposals(store)
        if not archived:
            print("No archived projects. `--new-project` archives the active one.")
            return 0
        print(f"{len(archived)} archived project(s):")
        for item in archived:
            print(f"  {item['key']}")
            print(f"    topic   : {item.get('thesis_topic', '')}")
            print(f"    question: {item.get('research_question', '')}")
        return 0

    return _with_store(report)


def restore_project(archive_key: str) -> int:
    from session import restore_archived_proposal

    def action(store) -> int:
        restored = restore_archived_proposal(store, archive_key)
        if restored is None:
            print(f"No archived project named {archive_key!r}. Run --list-projects to see them.")
            return 1
        print(f"Restored: {restored.get('thesis_topic', '')}")
        print("The previously active proposal was archived in its place.")
        return 0

    return _with_store(action)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Academic Research & Writing Assistant CLI",
        epilog=(
            "A thread is a conversation; a project is a thesis. The saved proposal outlives "
            "threads, so a new thread continues the same thesis. Use --new-project to start "
            "a different one."
        ),
    )
    parser.add_argument(
        "--new-thread",
        "--new",
        dest="new_thread",
        action="store_true",
        help="Start a fresh conversation thread on the SAME thesis. (--new is an alias.)",
    )
    parser.add_argument(
        "--new-project",
        dest="new_project",
        action="store_true",
        help="Start a new thesis: archives the current proposal and proposes a new one.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print the active thesis, thread, constraints and models, then exit.",
    )
    parser.add_argument(
        "--list-projects",
        dest="list_projects",
        action="store_true",
        help="List archived thesis proposals and exit.",
    )
    parser.add_argument(
        "--restore-project",
        dest="restore_project",
        metavar="KEY",
        help="Make an archived proposal active again (archives the current one first).",
    )
    parser.add_argument(
        "--format",
        dest="export_format",
        default="md",
        choices=("md", "tex", "docx", "pdf"),
        help="Format for exported drafts (default: md).",
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    if args.status:
        raise SystemExit(print_status())
    if args.list_projects:
        raise SystemExit(print_projects())
    if args.restore_project:
        raise SystemExit(restore_project(args.restore_project))
    run_cli(
        new_thread=args.new_thread,
        new_project=args.new_project,
        export_format=args.export_format,
    )
