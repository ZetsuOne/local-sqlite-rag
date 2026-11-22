from __future__ import annotations

import argparse
import hashlib
import math
import re
import array
from pathlib import Path

from openai import OpenAI

from db import connect
from parsers import extract_text

SUPPORTED = {".fb2", ".epub", ".pdf", ".txt", ".md", ".zip"}


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def series_for(path: Path, docs_root: Path) -> str | None:
    try:
        rel = path.relative_to(docs_root)
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) < 2:
        return None
    return parts[0]


def split_sentences(paragraph: str) -> list[str]:
    parts = re.split(r"(?<=[.!?…])\s+", paragraph.strip())
    return [p for p in parts if p.strip()]


def chunk_text(text: str, chunk_size: int = 1600, overlap: int = 250) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units: list[str] = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            units.append(para)
        else:
            units.extend(split_sentences(para) or [para])

    if not units:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if current:
            chunks.append("\n\n".join(current).strip())
            current = []
            current_len = 0

    for unit in units:
        add_len = len(unit) + (2 if current else 0)
        if current and current_len + add_len > chunk_size:
            flush()
            if overlap > 0 and chunks:
                prev = chunks[-1]
                tail = prev[-overlap:] if len(prev) > overlap else prev
                current = [tail]
                current_len = len(tail)
        if current:
            current.append(unit)
            current_len += add_len
        else:
            current = [unit]
            current_len = len(unit)
        if current_len >= chunk_size:
            flush()

    flush()
    return [c for c in chunks if c.strip()]


def l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def quantize_int8(vec: list[float]) -> bytes:
    normalized = l2_normalize(vec)
    return array.array(
        "b",
        [max(-127, min(127, int(round(x * 127.0)))) for x in normalized],
    ).tobytes()


def embed_batch(client: OpenAI, model: str, texts: list[str]) -> list[bytes]:
    resp = client.embeddings.create(model=model, input=texts)
    by_index = {item.index: item.embedding for item in resp.data}
    return [quantize_int8(by_index[i]) for i in range(len(texts))]


def iter_docs(docs_root: Path):
    for path in sorted(docs_root.rglob("*")):
        if not path.is_file():
            continue
        name = path.name.lower()
        if name.endswith(".fb2.zip") or path.suffix.lower() in SUPPORTED - {".zip"}:
            yield path
        elif path.suffix.lower() == ".zip" and ".fb2" in name:
            yield path


def delete_file_record(conn, file_id: int) -> None:
    chunk_ids = [
        r["id"]
        for r in conn.execute(
            "SELECT id FROM chunks WHERE file_id = ?", (file_id,)
        ).fetchall()
    ]
    if chunk_ids:
        placeholders = ",".join("?" * len(chunk_ids))
        conn.execute(
            f"DELETE FROM chunks_vec WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        )
    conn.execute("DELETE FROM files WHERE id = ?", (file_id,))


def index_file(
    conn,
    client: OpenAI,
    path: Path,
    docs_root: Path,
    model: str,
    batch_size: int,
    chunk_size: int,
    overlap: int,
) -> str:
    rel = str(path.relative_to(docs_root))
    digest = file_hash(path)

    existing = conn.execute(
        "SELECT id, file_hash FROM files WHERE path = ?", (rel,)
    ).fetchone()
    if existing and existing["file_hash"] == digest:
        return "skip"

    if existing:
        delete_file_record(conn, existing["id"])

    title, text = extract_text(path)
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        return "empty"

    cur = conn.execute(
        "INSERT INTO files(path, series, title, file_hash) VALUES (?, ?, ?, ?)",
        (rel, series_for(path, docs_root), title, digest),
    )
    file_id = cur.lastrowid

    chunk_ids: list[int] = []
    for i, content in enumerate(chunks):
        cur = conn.execute(
            "INSERT INTO chunks(file_id, chunk_index, content) VALUES (?, ?, ?)",
            (file_id, i, content),
        )
        chunk_ids.append(cur.lastrowid)

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors = embed_batch(client, model, batch)
        rows = [
            (chunk_ids[start + i], vectors[i])
            for i in range(len(batch))
        ]
        conn.executemany(
            "INSERT INTO chunks_vec(chunk_id, embedding) VALUES (?, vec_int8(?))",
            rows,
        )

    conn.commit()
    return f"ok:{len(chunks)}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into SQLite RAG")
    parser.add_argument("--docs", required=True, type=Path)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--base-url", default="http://localhost:1234/v1")
    parser.add_argument("--model", default="text-embedding-bge-m3")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=1600)
    parser.add_argument("--overlap", type=int, default=250)
    args = parser.parse_args()

    docs_root = args.docs.resolve()
    if not docs_root.is_dir():
        raise SystemExit(f"Docs folder not found: {docs_root}")

    client = OpenAI(base_url=args.base_url, api_key="lm-studio")
    conn = connect(args.db)

    files = list(iter_docs(docs_root))
    total = len(files)
    print(f"Found {total} files in {docs_root}")

    for i, path in enumerate(files, 1):
        rel = path.relative_to(docs_root)
        try:
            status = index_file(
                conn,
                client,
                path,
                docs_root,
                args.model,
                args.batch_size,
                args.chunk_size,
                args.overlap,
            )
            print(f"[{i}/{total}] {rel} -> {status}")
        except Exception as exc:
            conn.rollback()
            print(f"[{i}/{total}] {rel} -> error: {exc}")

    conn.close()


if __name__ == "__main__":
    main()
