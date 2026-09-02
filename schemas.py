from typing import Literal
from pydantic import BaseModel, Field

DecisionCategory = Literal[
    "thesis_decision",         # General thesis-level decisions
    "research_direction",      # Accepted or rejected paths
    "research_conclusion",     # Established empirical/theoretical findings
    "writing_preference",      # Style, citations, formatting, tone
    "supervisor_feedback",     # Explicit supervisor advice or demands
    "constraint",              # Hardware limits, datasets, deadlines, length
    "methodology_decision"     # Models, algorithms, baselines selected
]

class DurableMemoryItem(BaseModel):
    category: DecisionCategory = Field(description="The exact category of the durable memory.")
    title: str = Field(description="Short, specific summary (e.g., 'Reject Word2Vec, enforce ModernBERT').")
    content: str = Field(description="The full context, rationale, or directive to follow.")
    status: Literal["accepted", "rejected", "enforced"] = Field(
        default="enforced",
        description="Whether this direction was accepted, rejected, or must be strictly enforced."
    )
    
class ThesisProposal(BaseModel):
    thesis_topic: str = Field(description="The formal academic title/topic of the thesis.")
    research_question: str = Field(description="The primary core research question or hypothesis to be investigated.")
    outline: list[str] = Field(description="A sequential list of thesis chapters, sections, or analytical points to cover.")

class SupervisorDecision(BaseModel):
    next_step: Literal["research_node", "write_node"] = Field(
        description="Choose 'research_node' if the task requires web search, files, or execution. Choose 'write_node' for direct drafting/editing."
    )

class ReviewDecision(BaseModel):
    next_step: Literal["approved", "needs_more_research", "needs_revision"] = Field(
        description="Select 'approved' to finish, 'needs_more_research' if data/facts are missing, or 'needs_revision' for writing/style fixes."
    )
    issues: list[str] = Field(
        default_factory=list,
        description="List of specific flaws, factual gaps, or stylistic issues identified in the draft."
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description="Actionable, step-by-step suggestions explaining what the next worker should research or rewrite."
    )

class GeneratedCode(BaseModel):
    code: str = Field(description="Raw executable Python code with no markdown formatting.")