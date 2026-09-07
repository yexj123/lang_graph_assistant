"""Chunk and embed the PDFs in ./papers/ into the pgvector index.

Ingestion is idempotent. Chunk ids are a hash of (source, page, text), and PGVector
upserts on id conflict, so re-running re-embeds but does not duplicate. Previously every
run appended the whole corpus again under fresh UUIDs, which quietly doubled the index
and skewed top-k retrieval - duplicate chunks crowd out the results they displace.
"""

import argparse
import hashlib
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import get_vector_store, startup_error_message

PAPERS_DIR = Path("./papers")


def chunk_id(chunk: Document) -> str:
    """Stable id for a chunk, so re-ingesting the same text updates it in place.

    Includes source and page as well as the text: identical boilerplate (a running
    header, say) can appear on many pages, and collapsing those into one row would
    silently drop real chunks from the index.
    """
    payload = "|".join(
        (
            str(chunk.metadata.get("source", "")),
            str(chunk.metadata.get("page", "")),
            chunk.page_content,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_ingestion(rebuild: bool = False) -> None:
    if not PAPERS_DIR.exists():
        PAPERS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {PAPERS_DIR.resolve()}")
        print("Please place your PDF papers inside this directory and rerun.")
        return

    pdf_files = list(PAPERS_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {PAPERS_DIR.resolve()}.")
        print("Add PDFs, or run `python fetch_papers.py` for the reference corpus, then rerun.")
        return

    print(f"Found {len(pdf_files)} PDF(s) to process.")

    # Chunking strategy suitable for academic literature
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )

    all_chunks = []
    for pdf_path in pdf_files:
        print(f"-> Loading: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        chunks = splitter.split_documents(docs)

        # Ensure filename and page metadata are attached for agent citation
        for chunk in chunks:
            chunk.metadata["source"] = pdf_path.name

        all_chunks.extend(chunks)

    ids = [chunk_id(chunk) for chunk in all_chunks]
    distinct = len(set(ids))
    if distinct != len(ids):
        # Upsert collapses these, so say so rather than letting the index quietly shrink.
        print(f"Note: {len(ids) - distinct} chunk(s) are byte-identical and will collapse into one row each.")

    print(f"Generated {len(all_chunks)} chunks. Generating embeddings and storing in pgvector...")
    try:
        vector_store = get_vector_store()
        if rebuild:
            # Upsert cannot remove chunks whose source PDF is gone; only a rebuild can.
            print("Rebuilding: dropping the existing collection first.")
            vector_store.delete_collection()
            vector_store.create_collection()
        vector_store.add_documents(all_chunks, ids=ids)
    except Exception as exc:
        raise SystemExit(startup_error_message(exc)) from exc

    print(f"Ingestion complete ({distinct} distinct chunks). Papers are now searchable by the agent.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Embed ./papers/ into the pgvector index.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Drop the collection before ingesting. Use after removing or renaming a PDF, "
             "since an upsert alone leaves the old chunks behind.",
    )
    return parser


if __name__ == "__main__":
    run_ingestion(rebuild=build_arg_parser().parse_args().rebuild)
