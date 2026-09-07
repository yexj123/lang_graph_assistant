import operator
from typing import Annotated, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage

class ThesisState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    user_query: str

    # Thesis domain ground truth
    thesis_initialized: bool
    thesis_topic: str
    research_question: str
    outline: list[str]
    proposal_approved: bool   # gated by proposal_approval_node before research spends anything
    proposal_feedback: list[str]

    # Workspace artifacts
    research_notes: Annotated[list[str], operator.add]
    draft: str
    draft_path: str          # filesystem path of the last export, "" if never exported

    # Review loop & HITL
    request_type: str
    review_status: str
    review_feedback: list[str]
    revision_count: int