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
  • Any other text      -> Treated as revision feedback."""


def run_cli(new_thread: bool = False, new_project: bool = False) -> None:
    # Deferred so that --help, and any import of this module, stay free of I/O.
    from langchain_core.messages import HumanMessage
    from langgraph.types import Command

    from config import get_pool, get_store, init_db, startup_error_message
    from graph import build_graph
    from session import archive_active_proposal, resolve_thread_id

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
                    _service_interrupts(graph, config, Command)
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
                    "research_notes": [],
                    "draft": "",
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
                _run_turn(graph, config, input_state, Command)
            except KeyboardInterrupt:
                print("\n[interrupted] Turn abandoned. The thread is checkpointed, nothing is lost.")
            except Exception as exc:  # noqa: BLE001 - a bad turn must not end the session
                print(f"\n[error] This turn failed: {type(exc).__name__}: {exc}")
                print("The thread is checkpointed, so your work is intact. Try again or rephrase.")


def _run_turn(graph, config: dict, input_state: dict, Command) -> None:
    """Run one graph invocation, then service any human-approval interrupts it raises."""
    graph.invoke(input_state, config=config)
    _service_interrupts(graph, config, Command)


def _service_interrupts(graph, config: dict, Command) -> None:
    """Drive the human-approval gate until the graph stops asking.

    Split out from _run_turn because a thread can already be sitting at the gate when
    the CLI starts, and that has to be handled before prompting for anything new.
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

        graph.invoke(Command(resume=user_action), config=config)
        state_snapshot = graph.get_state(config)


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
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run_cli(new_thread=args.new_thread, new_project=args.new_project)
