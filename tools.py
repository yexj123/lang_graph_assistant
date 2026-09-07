from pathlib import Path
from pypdf import PdfReader
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import SystemMessage, HumanMessage
from functools import lru_cache

from config import get_model, get_vector_store
from sandbox import run_sandboxed
from schemas import GeneratedCode

def format_citation(metadata: dict) -> str:
    """Render chunk metadata as a citable reference, e.g. 'attention.pdf, p.4'.

    ingest.py attaches the filename and PyPDFLoader attaches `page`, which is
    0-indexed; citations are 1-indexed, hence the +1. Falls back to the bare
    source when a chunk carries no usable page number.
    """
    source = metadata.get("source", "Unknown")
    page = metadata.get("page")
    if isinstance(page, int) and not isinstance(page, bool):
        return f"{source}, p.{page + 1}"
    return source


@tool
def search_thesis_literature(query: str, k: int = 4) -> str:
    """Searches indexed academic literature, thesis papers, and technical PDFs.
    Use this first to find ground-truth facts, methodologies, and benchmarks before web searching.
    Each result is prefixed with its SOURCE as 'filename, p.N' - carry that marker into your
    notes verbatim, so the written draft can cite the exact page a claim came from.
    """
    try:
        docs = get_vector_store().similarity_search(query, k=k)
        if not docs:
            return "No matching sections found in local literature collection."

        formatted = []
        for i, doc in enumerate(docs, 1):
            formatted.append(f"[{i}] SOURCE: {format_citation(doc.metadata)}\n{doc.page_content.strip()}")

        return "\n\n".join(formatted)
    except Exception as e:
        return f"Literature retrieval failed: {e}"
    
@tool
def web_search(query: str) -> str:
    """Searches the web for up-to-date information, papers, or documentation."""
    try:
        search = DuckDuckGoSearchRun()
        return search.invoke(query)
    except Exception as e:
        return f"Search error: {e}"

@tool
def file_reader(file_path: str) -> str:
    """Reads the content of a local file (supports .txt, .md, .py, .json, and .pdf)."""
    path = Path(file_path)
    if not path.exists():
        return f"Error: File '{file_path}' does not exist."
    try:
        if path.suffix.lower() == ".pdf":
            reader = PdfReader(file_path)
            return "\n".join([page.extract_text() or "" for page in reader.pages])
        return path.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading '{file_path}': {e}"

@tool
def code_generator(prompt: str) -> str:
    """Generates clean, runnable Python code based on a task description (typically a math or statistical task)."""
    code_model = get_model().with_structured_output(GeneratedCode)
    
    messages = [
        SystemMessage(
            content=(
                "You are an expert Python developer. Write clean, complete, executable Python code. "
                "Always make sure to print results to stdout using print()."
            )
        ),
        HumanMessage(content=f"Task: {prompt}"),
    ]
    
    result: GeneratedCode = code_model.invoke(messages)
    return result.code

@tool
def python_executor(code: str) -> str:
    """Executes arbitrary Python code in an isolated subprocess (scrubbed environment, no
    inherited secrets, 15s timeout) and returns the stdout or error traceback."""
    return run_sandboxed(code)

tools = [search_thesis_literature, web_search, file_reader, code_generator, python_executor]


@lru_cache(maxsize=1)
def get_model_with_tools():
    """The research agent's model. Lazy so importing tools.py stays free of I/O."""
    return get_model().bind_tools(tools)
