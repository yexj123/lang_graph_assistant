"""Download the reference corpus described by papers/manifest.json.

The PDFs are deliberately not committed: arXiv licences vary per paper and
redistribution is the author's call, not this repo's. Shipping the manifest plus
this script keeps the corpus exactly reproducible without republishing anyone's
work - `papers/*.pdf` stays gitignored, and the sha256 in the manifest is what
makes "the same corpus" verifiable rather than merely claimed.

    python fetch_papers.py            # download anything missing, verify checksums
    python fetch_papers.py --list     # show the corpus without downloading
    python fetch_papers.py --force    # re-download even if present
"""

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

PAPERS_DIR = Path(__file__).parent / "papers"
MANIFEST_PATH = PAPERS_DIR / "manifest.json"

# arXiv asks automated clients to identify themselves and to pace requests.
_USER_AGENT = "thesis-assistant-corpus-fetcher/1.0 (+https://github.com/yexj123/lang_graph_assistant)"
_DELAY_SECONDS = 3
_CHUNK = 64 * 1024


def load_manifest() -> list[dict]:
    """Read the corpus definition. Fails loudly rather than downloading nothing."""
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"Manifest not found: {MANIFEST_PATH}")
    entries = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["papers"]
    if not entries:
        raise SystemExit(f"Manifest is empty: {MANIFEST_PATH}")
    return entries


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def pdf_url(entry: dict) -> str:
    return f"https://arxiv.org/pdf/{entry['arxiv_id']}"


def download(entry: dict, destination: Path) -> None:
    request = urllib.request.Request(pdf_url(entry), headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        destination.write_bytes(response.read())


def fetch_all(force: bool = False, write_checksums: bool = False) -> int:
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    entries = load_manifest()
    failures = 0
    downloaded = 0

    for entry in entries:
        target = PAPERS_DIR / entry["filename"]

        if target.exists() and not force:
            actual = sha256_of(target)
            expected = entry.get("sha256")
            if expected and actual != expected:
                print(f"  [CORRUPT] {entry['filename']}: checksum mismatch, re-downloading")
            else:
                print(f"  [have]    {entry['filename']}")
                continue

        if downloaded:
            time.sleep(_DELAY_SECONDS)

        print(f"  [get]     {entry['filename']}  <- {pdf_url(entry)}")
        try:
            download(entry, target)
            downloaded += 1
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"  [FAILED]  {entry['filename']}: {type(exc).__name__}: {exc}")
            failures += 1
            continue

        actual = sha256_of(target)
        expected = entry.get("sha256")
        if write_checksums:
            entry["sha256"] = actual
        elif expected and actual != expected:
            # arXiv reissues papers under the same id, so this is a real signal that
            # the corpus has drifted from the one the golden set was written against.
            print(f"  [WARN]    {entry['filename']}: sha256 {actual} != manifest {expected}")

    if write_checksums:
        MANIFEST_PATH.write_text(
            json.dumps({"papers": entries}, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nChecksums written to {MANIFEST_PATH}")

    if failures:
        print(f"\n{failures} paper(s) failed to download. Re-run to retry.")
    else:
        print(f"\nCorpus ready in {PAPERS_DIR.resolve()}. Next: python ingest.py")
    return 1 if failures else 0


def print_listing() -> int:
    for entry in load_manifest():
        present = "yes" if (PAPERS_DIR / entry["filename"]).exists() else "no"
        print(f"{entry['arxiv_id']:<12} present={present:<4} {entry['title']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch the reference corpus for the RAG index.")
    parser.add_argument("--list", action="store_true", help="List the corpus without downloading.")
    parser.add_argument("--force", action="store_true", help="Re-download even if the file exists.")
    parser.add_argument(
        "--write-checksums",
        action="store_true",
        help="Record the sha256 of each downloaded file back into the manifest (maintainer use).",
    )
    args = parser.parse_args()

    if args.list:
        return print_listing()
    return fetch_all(force=args.force, write_checksums=args.write_checksums)


if __name__ == "__main__":
    raise SystemExit(main())
