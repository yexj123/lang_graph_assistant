from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode

from state import ThesisState
from tools import tools
from nodes import (
    proposal_node,
    proposal_approval_node,
    research_node,
    write_node,
    reviewer_node,
    human_approval_node,
    supervisor_router,
    research_router,
)
from config import get_checkpointer, get_store


def build_graph():
    """Compile the thesis graph.

    Assumes the tables already exist: run config.init_db() once before calling this.
    Migration is a separate responsibility and used to run here as an import-time side
    effect, which meant importing this module required a live database.
    """
    builder = StateGraph(ThesisState)

    # Register Nodes
    builder.add_node("proposal_node", proposal_node)
    builder.add_node("proposal_approval_node", proposal_approval_node)
    builder.add_node("research_node", research_node)
    builder.add_node("tool_node", ToolNode(tools))
    builder.add_node("write_node", write_node)
    builder.add_node("reviewer_node", reviewer_node)
    builder.add_node("human_approval_node", human_approval_node)

    # Connect Edges
    builder.add_edge(START, "proposal_node")

    # The plan is gated before any research is billed. proposal_approval_node calls
    # supervisor_router itself and routes with Command(goto=...), so there is no
    # conditional edge here - mixing the two from one node fights the framework.
    builder.add_edge("proposal_node", "proposal_approval_node")

    builder.add_conditional_edges(
        "research_node",
        research_router,
        {"tool_node": "tool_node", "write_node": "write_node"}
    )

    builder.add_edge("tool_node", "research_node")
    builder.add_edge("write_node", "reviewer_node")

    # proposal_approval_node, reviewer_node and human_approval_node use Command(goto=...)

    return builder.compile(checkpointer=get_checkpointer(), store=get_store())
