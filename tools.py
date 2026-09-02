import io
import sys
import traceback
from pathlib import Path
from pypdf import PdfReader
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import SystemMessage, HumanMessage

from config import model
from schemas import GeneratedCode

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
    code_model = model.with_structured_output(GeneratedCode)
    
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
    """Executes arbitrary Python code and returns the stdout or error traceback."""
    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    sys.stdout = redirected_output
    exec_globals = {}
    try:
        exec(code, exec_globals)
        output = redirected_output.getvalue()
        return output.strip() if output.strip() else "Execution successful (no output)."
    except Exception:
        return f"Execution Error:\n{traceback.format_exc()}"
    finally:
        sys.stdout = old_stdout

tools = [web_search, file_reader, code_generator, python_executor]
model_with_tools = model.bind_tools(tools)