from typing import Literal
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt, Command
from langgraph.graph import END
from langgraph.store.base import BaseStore
from memory import save_durable_memory, get_durable_memories, DurableMemoryItem

from config import model
from state import ThesisState
from schemas import ThesisProposal, SupervisorDecision, ReviewDecision
from tools import model_with_tools

# --- Routers ---
def supervisor_router(state: ThesisState) -> str:
    router_model = model.with_structured_output(SupervisorDecision)
    decision = router_model.invoke(f"Route this user query to the best worker:\n{state['user_query']}")
    return decision.next_step

def research_router(state: ThesisState) -> Literal["tool_node", "write_node"]:
    messages = state.get("messages", [])
    if messages and getattr(messages[-1], "tool_calls", None):
        return "tool_node"
    return "write_node"

# --- Nodes ---
def proposal_node(state: ThesisState, store: BaseStore) -> dict:
    if state.get("thesis_initialized", False):
        return {}

    namespace = ("thesis_workspace", "active_project")
    key = "current_proposal"

    saved_proposal = store.get(namespace, key)
    if saved_proposal and not state.get("thesis_topic"):
        data = saved_proposal.value
        return {
            "thesis_initialized": True,
            "thesis_topic": data["thesis_topic"],
            "research_question": data["research_question"],
            "outline": data["outline"],
        }

    proposal_model = model.with_structured_output(ThesisProposal)
    system_prompt = (
        "You are an academic thesis advisor.\n\n"
        "Based on the user's initial thesis idea, formulate:\n"
        "1. A clear and academically appropriate thesis topic.\n"
        "2. A concrete primary research question.\n"
        "3. A logical sequential thesis outline."
    )

    proposal: ThesisProposal = proposal_model.invoke(
        f"{system_prompt}\n\nUSER'S INITIAL THESIS IDEA:\n{state['user_query']}"
    )

    store.put(
        namespace,
        key,
        {
            "thesis_topic": proposal.thesis_topic,
            "research_question": proposal.research_question,
            "outline": proposal.outline,
        }
    )

    return {
        "thesis_initialized": True,
        "thesis_topic": proposal.thesis_topic,
        "research_question": proposal.research_question,
        "outline": proposal.outline,
        "review_feedback": [],
    }

def research_node(state: ThesisState, store: BaseStore) -> dict:
    durable_context = get_durable_memories(store)
    outline_str = "\n".join(f"- {sec}" for sec in state.get("outline", []))
    feedback_context = ""
    if state.get("review_feedback"):
        feedback_context = "\nAddress this reviewer feedback:\n" + "\n".join(
            f"- {f}" for f in state["review_feedback"]
        )

    system_prompt = SystemMessage(
        content=(
            "You are a graduate research assistant conducting rigorous technical investigation.\n\n"
            f"THESIS TOPIC: {state.get('thesis_topic')}\n"
            f"PRIMARY RESEARCH QUESTION: {state.get('research_question')}\n\n"
            f"ENFORCED DECISIONS & CONSTRAINTS:\n{durable_context}\n\n"
            f"TARGET OUTLINE:\n{outline_str}\n\n"
            "Use your tools to gather verified facts, empirical data, code implementations, or theoretical proofs "
            "for EACH point in the outline to decisively answer the research question. "
            "Synthesize your notes into structured academic findings."
            f"{feedback_context}"
        )
    )

    prompt_messages = [system_prompt] + state.get("messages", [])
    response = model_with_tools.invoke(prompt_messages)

    notes_update = {}
    if not getattr(response, "tool_calls", None) and response.content:
        notes_update["research_notes"] = [response.content]

    return {
        "messages": [response],
        **notes_update
    }

def write_node(state: ThesisState, store: BaseStore) -> dict:
    durable_context = get_durable_memories(store)
    outline_str = "\n".join(f"{i+1}. {sec}" for i, sec in enumerate(state.get("outline", [])))

    feedback_context = ""
    if state.get("review_feedback"):
        feedback_context = "\nAddress reviewer feedback:\n" + "\n".join(
            f"- {f}" for f in state["review_feedback"]
        )

    system_prompt = (
        "You are an academic thesis writer. Draft a formal academic thesis chapter/paper.\n\n"
        f"THESIS TOPIC: {state.get('thesis_topic')}\n"
        f"CORE RESEARCH QUESTION: {state.get('research_question')}\n\n"
        f"MANDATORY OUTLINE:\n{outline_str}\n\n"
        f"ENFORCED DECISIONS, CONSTRAINTS & SUPERVISOR GUIDELINES:\n{durable_context}\n\n"
        f"RESEARCH NOTES:\n" + "\n".join(f"- {note}" for note in state.get("research_notes", [])) + "\n"
        f"{feedback_context}"
    )

    response = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Draft the thesis document addressing: {state.get('research_question')}"),
    ])

    return {
        "draft": response.content,
        "messages": [response],
    }

def reviewer_node(state: ThesisState) -> Command[Literal["research_node", "write_node", "human_approval_node"]]:
    reviewer_model = model.with_structured_output(ReviewDecision)
    outline_str = "\n".join(f"- {section}" for section in state.get("outline", []))

    system_prompt = (
        "You are a strict academic thesis reviewer.\n\n"
        f"THESIS TOPIC:\n{state.get('thesis_topic', '')}\n\n"
        f"RESEARCH QUESTION:\n{state.get('research_question', '')}\n\n"
        f"REQUIRED OUTLINE:\n{outline_str}\n\n"
        f"CURRENT DRAFT:\n{state.get('draft', '')}\n"
    )

    review: ReviewDecision = reviewer_model.invoke(system_prompt)
    current_revisions = state.get("revision_count", 0) + 1
    combined_feedback = [f"[ACADEMIC ISSUE] {i}" for i in review.issues] + [
        f"[ACADEMIC SUGGESTION] {s}" for s in review.suggestions
    ]

    if review.next_step == "approved" or current_revisions >= 3:
        return Command(
            update={
                "review_status": "ai_approved" if review.next_step == "approved" else "max_revisions_reached",
                "review_feedback": combined_feedback,
                "revision_count": current_revisions,
            },
            goto="human_approval_node",
        )

    if review.next_step == "needs_more_research":
        return Command(
            update={
                "review_status": "needs_more_research",
                "review_feedback": combined_feedback,
                "revision_count": current_revisions,
            },
            goto="research_node",
        )

    return Command(
        update={
            "review_status": "needs_revision",
            "review_feedback": combined_feedback,
            "revision_count": current_revisions,
        },
        goto="write_node",
    )

def human_approval_node(state: ThesisState, store: BaseStore) -> Command[Literal["research_node", "write_node", "__end__"]]:
    payload = {
        "current_draft": state.get("draft", ""),
        "review_status": state.get("review_status", ""),
        "review_feedback": state.get("review_feedback", []),
        "revision_count": state.get("revision_count", 0),
        "active_constraints": get_durable_memories(store),
    }
    
    human_response = interrupt(payload)
    response_str = str(human_response).strip()
    
    if response_str.lower().startswith("remember:"):
        raw_text = response_str[len("remember:"):].strip()
        
        extractor = model.with_structured_output(DurableMemoryItem)
        memory_item: DurableMemoryItem = extractor.invoke(
            f"Classify and structure this durable thesis memory into its category:\n{raw_text}"
        )
        save_durable_memory(store, memory_item)

        return Command(
            update={"review_feedback": [f"[LONG-TERM MEMORY SAVED]: {memory_item.title}"],
                    "revision_count": 0},
            goto="write_node",
        )

    # 2. Final Approval -> Automatically record approved conclusions
    if response_str.lower() in {"approve", "approved", "ok", "yes", "looks good"}:
        notes_text = "\n".join(state.get("research_notes", []))[:400]
        conclusion = DurableMemoryItem(
            category="research_conclusion",
            title=f"Validated section for '{state.get('research_question')}'",
            content = notes_text,
            status="accepted"
        )
        save_durable_memory(store, conclusion)
        return Command(update={"review_status": "final_approved"}, goto=END)

    # 3. Explicit Research Override (also save direction if user specified rejection)
    if response_str.lower().startswith("research:"):
        feedback = response_str[len("research:"):].strip()
        return Command(
            update={
                "review_status": "needs_more_research",
                "review_feedback": [f"[HUMAN FEEDBACK - RESEARCH]: {feedback}"],
                "revision_count": 0,
            },
            goto="research_node",
        )

    # 4. Revisions
    if response_str.lower().startswith("revise:"):
        feedback = response_str[len("revise:"):].strip()
    else:
        feedback = response_str

    return Command(
        update={
            "review_status": "needs_revision",
            "review_feedback": [f"[HUMAN FEEDBACK - WRITING]: {feedback}"],
            "revision_count": 0,
        },
        goto="write_node",
    )