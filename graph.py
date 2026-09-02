from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode

from state import ThesisState
from tools import tools
from nodes import (
    proposal_node,
    research_node,
    write_node,
    reviewer_node,
    human_approval_node,
    supervisor_router,
    research_router,
)
from config import checkpointer, store, init_db

def build_graph():
    # Run migrations
    init_db()

    builder = StateGraph(ThesisState)

    # Register Nodes
    builder.add_node("proposal_node", proposal_node)
    builder.add_node("research_node", research_node)
    builder.add_node("tool_node", ToolNode(tools))
    builder.add_node("write_node", write_node)
    builder.add_node("reviewer_node", reviewer_node)
    builder.add_node("human_approval_node", human_approval_node)

    # Connect Edges
    builder.add_edge(START, "proposal_node")

    builder.add_conditional_edges(
        "proposal_node",
        supervisor_router,
        {"research_node": "research_node", "write_node": "write_node"}
    )

    builder.add_conditional_edges(
        "research_node",
        research_router,
        {"tool_node": "tool_node", "write_node": "write_node"}
    )

    builder.add_edge("tool_node", "research_node")
    builder.add_edge("write_node", "reviewer_node")

    # reviewer_node and human_approval_node use Command(goto=...)

    return builder.compile(checkpointer=checkpointer, store=store)

graph = build_graph()