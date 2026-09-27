"""Load only local corpus files and create section-aware retrieval chunks."""
from __future__ import annotations

import re
from pathlib import Path

from index import chunk_document

SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown", ".pdf"}


def _safe_id(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.-")
    return value or "document"


def _read_markdown(path: Path) -> list[tuple[str, str]]:
    sections: list[tuple[str, list[str]]] = []
    current_title = "1"
    current_lines: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", line)
        if match:
            if any(part.strip() for part in current_lines):
                sections.append((current_title, current_lines))
            current_title, current_lines = match.group(1).strip(), []
        else:
            current_lines.append(line)
    if any(part.strip() for part in current_lines):
        sections.append((current_title, current_lines))
    if not sections:
        return [("1", path.read_text(encoding="utf-8", errors="replace"))]
    return [(str(i + 1), "\n".join(lines)) for i, (_, lines) in enumerate(sections)]


def _read_file(path: Path) -> list[tuple[str, str]]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".markdown"}:
        if suffix == ".txt":
            return [("1", path.read_text(encoding="utf-8", errors="replace"))]
        return _read_markdown(path)
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("Reading PDFs requires pypdf; install the project requirements.") from exc
        reader = PdfReader(str(path))
        return [(str(i + 1), page.extract_text() or "") for i, page in enumerate(reader.pages)]
    return []


def load_corpus(directory: str | Path, max_words: int = 180, overlap: int = 30):
    """Recursively load local TXT, Markdown, and text-based PDF files.

    Citations use the source filename as Doc_ID and a one-based section
    number (Markdown heading or PDF page) as Section. No network is used.
    """
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Corpus directory does not exist: {root}")
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)
    if not paths:
        raise ValueError(f"No supported .txt, .md, .markdown, or .pdf files found in {root}")

    chunks = []
    seen: set[str] = set()
    for path in paths:
        relative = path.relative_to(root).with_suffix("").as_posix()
        doc_id = _safe_id(relative.replace("/", "_"))
        if doc_id in seen:
            raise ValueError(f"Corpus files produce duplicate Doc_ID {doc_id!r}")
        seen.add(doc_id)
        for section, body in _read_file(path):
            meta = {"source": path.relative_to(root).as_posix(), "section": section}
            chunks.extend(chunk_document(doc_id, body, max_words=max_words, overlap=overlap, meta=meta))
    if not chunks:
        raise ValueError(f"Corpus files under {root} contained no extractable text")
    return chunks
