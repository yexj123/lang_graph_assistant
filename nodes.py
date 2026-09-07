from typing import Literal
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt, Command
from langgraph.graph import END
from langgraph.store.base import BaseStore
from memory import (
    DurableMemoryItem,
    forget_durable_memory,
    format_memory_listing,
    get_durable_memories,
    list_durable_memories,
    save_durable_memory,
)
from session import PROPOSAL_NAMESPACE, PROPOSAL_KEY
from hitl import parse_human_decision
import export
from export import write_draft
from context import condense_notes, trim_history

from config import get_model
from state import ThesisState
from schemas import ThesisProposal, SupervisorDecision, ReviewDecision
from tools import get_model_with_tools

# --- Routers ---
def supervisor_router(state: ThesisState) -> str:
    # .get, not ['user_query']: this now also runs from proposal_approval_node, and a
    # KeyError inside a router aborts the whole turn rather than degrading.
    router_model = get_model().with_structured_output(SupervisorDecision)
    decision = router_model.invoke(
        f"Route this user query to the best worker:\n{state.get('user_query', '')}"
    )
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

    namespace, key = PROPOSAL_NAMESPACE, PROPOSAL_KEY

    saved_proposal = store.get(namespace, key)
    if saved_proposal and not state.get("thesis_topic") and not state.get("proposal_feedback"):
        data = saved_proposal.value
        return {
            "thesis_initialized": True,
            # A stored proposal was approved in an earlier session; don't re-gate it.
            "proposal_approved": True,
            "thesis_topic": data["thesis_topic"],
            "research_question": data["research_question"],
            "outline": data["outline"],
        }

    proposal_model = get_model().with_structured_output(ThesisProposal)

    revision = ""
    if state.get("proposal_feedback"):
        revision = "\n\nThe human rejected the previous proposal. Address this feedback:\n" + "\n".join(
            f"- {f}" for f in state["proposal_feedback"]
        )

    system_prompt = (
        "You are an academic thesis advisor.\n\n"
        "Based on the user's initial thesis idea, formulate:\n"
        "1. A clear and academically appropriate thesis topic.\n"
        "2. A concrete primary research question.\n"
        "3. A logical sequential thesis outline."
        + revision
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
        "proposal_approved": False,
        "proposal_feedback": [],
        "thesis_topic": proposal.thesis_topic,
        "research_question": proposal.research_question,
        "outline": proposal.outline,
        "review_feedback": [],
    }

def proposal_approval_node(
    state: ThesisState, store: BaseStore
) -> Command[Literal["proposal_node", "proposal_approval_node", "research_node", "write_node"]]:
    """Put the plan to the human before any research is paid for.

    The draft gate came far too late in the process: proposal_node invented the topic,
    question and outline and routed straight into research, so the first time anyone saw
    the plan was bundled inside a finished draft - after every research call had been
    billed. For a thesis, the plan is the expensive thing to get wrong.

    Approving here routes onward via supervisor_router; anything else regenerates the
    proposal with the human's feedback attached.
    """
    if state.get("proposal_approved"):
        return Command(goto=supervisor_router(state))

    outline = state.get("outline", [])
    decision = parse_human_decision(
        interrupt(
            {
                "kind": "proposal",
                "thesis_topic": state.get("thesis_topic", ""),
                "research_question": state.get("research_question", ""),
                "outline": outline,
                "active_constraints": get_durable_memories(store),
            }
        )
    )

    if decision.action == "empty":
        return Command(goto="proposal_approval_node")

    if decision.action == "approve":
        return Command(update={"proposal_approved": True}, goto=supervisor_router(state))

    # Everything else is a rewrite instruction for the plan, not for a draft.
    return Command(
        update={
            "proposal_approved": False,
            "proposal_feedback": [decision.payload],
            "thesis_initialized": False,
            "thesis_topic": "",
        },
        goto="proposal_node",
    )


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
            "Synthesize your notes into structured academic findings.\n\n"
            "Attribute every finding to where it came from. search_thesis_literature returns a "
            "SOURCE marker of the form 'filename, p.N' with each result: carry it verbatim into "
            "your notes as [filename, p.N]. Mark anything from web_search with its URL, and "
            "anything you reasoned out yourself as [unsourced]. The writer can only cite what "
            "you attribute here."
            f"{feedback_context}"
        )
    )

    # Bounded: this history carries every prior tool result and full draft.
    prompt_messages = [system_prompt] + trim_history(state.get("messages", []))
    response = get_model_with_tools().invoke(prompt_messages)

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
        f"RESEARCH NOTES:\n" + "\n".join(f"- {note}" for note in condense_notes(state.get("research_notes", []))) + "\n\n"
        "CITATIONS: cite every factual claim inline as [source, p.N], copying the marker from "
        "the research notes above. Never invent a citation or a page number - if a claim has no "
        "marker in the notes, either omit the claim or write it without one and expect the "
        "reviewer to flag it. Close the document with a References section listing each distinct "
        "source you cited."
        f"{feedback_context}"
    )

    response = get_model().invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Draft the thesis document addressing: {state.get('research_question')}"),
    ])

    return {
        "draft": response.content,
        "messages": [response],
    }

def reviewer_node(state: ThesisState) -> Command[Literal["research_node", "write_node", "human_approval_node"]]:
    reviewer_model = get_model().with_structured_output(ReviewDecision)
    outline_str = "\n".join(f"- {section}" for section in state.get("outline", []))

    # The reviewer is asked to return 'needs_more_research' for factual gaps, so it has to
    # see the evidence the draft was built from. Without these notes it can only judge prose
    # against an outline, and any factual verdict it returns is unfounded.
    notes = state.get("research_notes", [])
    notes_str = "\n".join(f"- {note}" for note in condense_notes(notes)) if notes else "(none gathered yet)"

    system_prompt = (
        "You are a strict academic thesis reviewer.\n\n"
        f"THESIS TOPIC:\n{state.get('thesis_topic', '')}\n\n"
        f"RESEARCH QUESTION:\n{state.get('research_question', '')}\n\n"
        f"REQUIRED OUTLINE:\n{outline_str}\n\n"
        f"SOURCE MATERIAL GATHERED BY THE RESEARCH AGENT:\n{notes_str}\n\n"
        f"CURRENT DRAFT:\n{state.get('draft', '')}\n\n"
        "Judge the draft against the source material above, not against your own knowledge. "
        "Treat any factual claim that the source material does not support as an issue, and "
        "return 'needs_more_research' when the material is too thin to support the draft. "
        "Flag claims that carry no [source, p.N] citation."
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

def _export_draft(state: ThesisState, out_dir: str = "") -> dict:
    """Best-effort write of the current draft to disk.

    Returns a state update either way. A failed export must never lose an approval or
    crash the gate - but it must not be silent either, so the failure goes into
    review_feedback where the CLI prints it.
    """
    try:
        path = write_draft(
            state.get("draft", ""),
            thesis_topic=state.get("thesis_topic", ""),
            research_question=state.get("research_question", ""),
            fmt=export.DEFAULT_FORMAT,
            **({"out_dir": out_dir} if out_dir else {}),
        )
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        return {"review_feedback": [f"[EXPORT FAILED] {type(exc).__name__}: {exc}"]}
    return {"draft_path": str(path)}


def human_approval_node(state: ThesisState, store: BaseStore) -> Command[Literal["research_node", "write_node", "human_approval_node", "__end__"]]:
    payload = {
        "current_draft": state.get("draft", ""),
        "review_status": state.get("review_status", ""),
        "review_feedback": state.get("review_feedback", []),
        "revision_count": state.get("revision_count", 0),
        "active_constraints": get_durable_memories(store),
        "draft_path": state.get("draft_path", ""),
    }

    decision = parse_human_decision(interrupt(payload))

    # 0. Nothing entered. Re-ask rather than spending a rewrite on empty feedback, which
    #    is what the old fallthrough did.
    if decision.action == "empty":
        return Command(goto="human_approval_node")

    # 0b. Snapshot the draft to disk without deciding anything, then ask again. Lets you
    #     keep a version before requesting changes that might make it worse.
    if decision.action == "export":
        return Command(update=_export_draft(state, decision.payload), goto="human_approval_node")

    # 0c. Inspect and prune durable memory. Constraints steer every future generation, so
    #     being unable to see or remove one was the sharpest edge left in the tool.
    if decision.action == "memories":
        listing = format_memory_listing(list_durable_memories(store))
        return Command(update={"review_feedback": [listing]}, goto="human_approval_node")

    if decision.action == "forget":
        try:
            index = int(decision.payload.strip())
        except ValueError:
            note = f"[FORGET FAILED] Expected a number from the `memories` listing, got {decision.payload!r}."
        else:
            removed = forget_durable_memory(store, index)
            note = (
                f"[CONSTRAINT REMOVED] {removed['title']}"
                if removed
                else f"[FORGET FAILED] No constraint numbered {index}. Run `memories` for the list."
            )
        return Command(update={"review_feedback": [note]}, goto="human_approval_node")

    # 1. Durable memory -> persist, then redraft under the new constraint.
    if decision.action == "remember":
        extractor = get_model().with_structured_output(DurableMemoryItem)
        memory_item: DurableMemoryItem = extractor.invoke(
            f"Classify and structure this durable thesis memory into its category:\n{decision.payload}"
        )
        save_durable_memory(store, memory_item)

        return Command(
            update={"review_feedback": [f"[LONG-TERM MEMORY SAVED]: {memory_item.title}"],
                    "revision_count": 0},
            goto="write_node",
        )

    # 2. Final Approval -> Automatically record approved conclusions
    if decision.action == "approve":
        notes_text = "\n".join(state.get("research_notes", []))[:400]
        conclusion = DurableMemoryItem(
            category="research_conclusion",
            title=f"Validated section for '{state.get('research_question')}'",
            content = notes_text,
            status="accepted"
        )
        save_durable_memory(store, conclusion)

        # The graph ends here, so this is the last chance to put the thesis somewhere the
        # operator can actually reach. Without it the finished draft survives only inside
        # a Postgres checkpoint and in terminal scrollback.
        update = {"review_status": "final_approved"}
        update.update(_export_draft(state))
        return Command(update=update, goto=END)

    # 3. Explicit Research Override (also save direction if user specified rejection)
    if decision.action == "research":
        return Command(
            update={
                "review_status": "needs_more_research",
                "review_feedback": [f"[HUMAN FEEDBACK - RESEARCH]: {decision.payload}"],
                "revision_count": 0,
            },
            goto="research_node",
        )

    # 4. Revisions - either an explicit `revise:` or unrecognised text treated as feedback.
    feedback = decision.payload

    return Command(
        update={
            "review_status": "needs_revision",
            "review_feedback": [f"[HUMAN FEEDBACK - WRITING]: {feedback}"],
            "revision_count": 0,
        },
        goto="write_node",
    )