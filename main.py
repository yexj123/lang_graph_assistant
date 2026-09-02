import uuid
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from config import pool
from graph import graph

def run_cli():
    print("=== Academic Research & Writing Assistant Initialized ===")
    print("Type 'exit' or 'quit' to end the session.\n")

    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}
    first_query = True

    with pool:
        while True:
            query = input("\nEnter your query: ").strip()
            if query.lower() in {"exit", "quit"}:
                print("Session terminated.")
                break
            if not query:
                continue

            if first_query:
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
                first_query = False
            else:
                input_state = {
                    "user_query": query,
                    "messages": [HumanMessage(content=query)],
                    "review_feedback": [],
                }

            result = graph.invoke(input_state, config=config)

            # Human-in-the-Loop handling
            state_snapshot = graph.get_state(config)
            while state_snapshot.next:
                interrupt_payload = state_snapshot.tasks[0].interrupts[0].value

                print("\n" + "=" * 60)
                print("                  HUMAN APPROVAL GATE")
                print("=" * 60)
                print(f"Reviewer Verdict: {interrupt_payload.get('review_status', 'N/A')}")

                feedback_items = interrupt_payload.get("review_feedback", [])
                if feedback_items:
                    print("\nFeedback:")
                    for item in feedback_items:
                        print(f"  {item}")

                # ---------------------------------------------------------
                # PRINT ACTIVE LONG-TERM CONSTRAINTS HERE
                # ---------------------------------------------------------
                active_constraints = interrupt_payload.get("active_constraints")
                if active_constraints and active_constraints != "No durable decisions recorded yet.":
                    print("\nActive Long-Term Constraints:")
                    print(active_constraints)

                print("\nCurrent Generated Draft:")
                print("-" * 60)
                print(interrupt_payload.get("current_draft", "No draft content available."))
                print("-" * 60)

                print("\nAction Commands:")
                print("  • 'approve'           -> Accept draft and finalize.")
                print("  • 'research: <notes>' -> Request more research.")
                print("  • 'revise: <notes>'   -> Request writing revision.")
                print("  • 'remember: <rule>'  -> Save constraint/decision to long-term memory.")
                print("  • Any other text      -> Treated as revision feedback.")

                user_action = input("\nYour decision: ").strip()

                result = graph.invoke(Command(resume=user_action), config=config)
                state_snapshot = graph.get_state(config)

if __name__ == "__main__":
    run_cli()