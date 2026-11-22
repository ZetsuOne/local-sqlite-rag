from __future__ import annotations
import argparse
import re
from collections import defaultdict
from openai import OpenAI
from db import connect
from ingest import quantize_int8
def escape_fts5(query: str) -> str:
    tokens = re.findall(r"\w+", query, flags=re.UNICODE)
    if not tokens:
        return '""'
    return " OR ".join(f'"{t}"' for t in tokens)
def embed_query(client: OpenAI, model: str, query: str) -> bytes:
    resp = client.embeddings.create(model=model, input=[query])
    return quantize_int8(resp.data[0].embedding)
def search_vec(conn, query_vec: bytes, series: str | None, limit: int = 20):
    if series:
        rows = conn.execute(
            """
            SELECT v.chunk_id, v.distance, c.content, f.title, f.series, f.path, c.chunk_index
            FROM chunks_vec v
            JOIN chunks c ON c.id = v.chunk_id
            JOIN files f ON f.id = c.file_id
            WHERE v.embedding MATCH vec_int8(?)
              AND k = ?
              AND f.series = ?
            ORDER BY v.distance
            """,
            (query_vec, limit, series),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT v.chunk_id, v.distance, c.content, f.title, f.series, f.path, c.chunk_index
            FROM chunks_vec v
            JOIN chunks c ON c.id = v.chunk_id
            JOIN files f ON f.id = c.file_id
            WHERE v.embedding MATCH vec_int8(?)
              AND k = ?
            ORDER BY v.distance
            """,
            (query_vec, limit),
        ).fetchall()
    return rows
def search_fts(conn, query: str, series: str | None, limit: int = 20):
    fts_query = escape_fts5(query)
    if series:
        rows = conn.execute(
            """
            SELECT c.id AS chunk_id, bm25(chunks_fts) AS rank,
                   c.content, f.title, f.series, f.path, c.chunk_index
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            JOIN files f ON f.id = c.file_id
            WHERE chunks_fts MATCH ?
              AND f.series = ?
            ORDER BY rank
            LIMIT ?
            """,
            (fts_query, series, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT c.id AS chunk_id, bm25(chunks_fts) AS rank,
                   c.content, f.title, f.series, f.path, c.chunk_index
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            JOIN files f ON f.id = c.file_id
            WHERE chunks_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (fts_query, limit),
        ).fetchall()
    return rows


def rrf_fuse(vec_rows, fts_rows, top_k: int, k: int = 60):
    scores: dict[int, float] = defaultdict(float)
    meta: dict[int, dict] = {}

    for rank, row in enumerate(vec_rows, 1):
        cid = row["chunk_id"]
        scores[cid] += 1.0 / (k + rank)
        meta[cid] = {
            "content": row["content"],
            "title": row["title"],
            "series": row["series"],
            "path": row["path"],
            "chunk_index": row["chunk_index"],
        }

    for rank, row in enumerate(fts_rows, 1):
        cid = row["chunk_id"]
        scores[cid] += 1.0 / (k + rank)
        if cid not in meta:
            meta[cid] = {
                "content": row["content"],
                "title": row["title"],
                "series": row["series"],
                "path": row["path"],
                "chunk_index": row["chunk_index"],
            }

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    results = []
    for cid, score in ranked:
        item = dict(meta[cid])
        item["chunk_id"] = cid
        item["score"] = score
        results.append(item)
    return results


def answer_with_llm(client: OpenAI, model: str, query: str, hits: list[dict]) -> str:
    context_parts = []
    for i, hit in enumerate(hits, 1):
        series = hit["series"] or "—"
        context_parts.append(
            f"[{i}] {hit['title']} (series: {series}, path: {hit['path']})\n{hit['content']}"
        )
    context = "\n\n".join(context_parts)

    messages = [
        {
            "role": "system",
            "content": (
                "Отвечай на вопрос пользователя, опираясь только на переданный контекст. "
                "В конце укажи ссылки на книги в формате [N] Название."
            ),
        },
        {
            "role": "user",
            "content": f"Контекст:\n{context}\n\nВопрос: {query}",
        },
    ]
    resp = client.chat.completions.create(model=model, messages=messages)
    return resp.choices[0].message.content or ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid search over SQLite RAG")
    parser.add_argument("--db", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--series", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--base-url", default="http://localhost:1234/v1")
    parser.add_argument("--embed-model", default="text-embedding-bge-m3")
    parser.add_argument("--chat-model", default=None)
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="lm-studio")
    conn = connect(args.db)

    query_vec = embed_query(client, args.embed_model, args.query)
    vec_rows = search_vec(conn, query_vec, args.series, limit=20)
    fts_rows = search_fts(conn, args.query, args.series, limit=20)
    hits = rrf_fuse(vec_rows, fts_rows, top_k=args.top_k)

    if not hits:
        print("No results.")
        conn.close()
        return

    if args.chat_model:
        answer = answer_with_llm(client, args.chat_model, args.query, hits)
        print(answer)
        print("\nSources:")
        for i, hit in enumerate(hits, 1):
            series = hit["series"] or "—"
            print(
                f"  [{i}] {hit['title']} | series={series} | "
                f"chunk={hit['chunk_index']} | score={hit['score']:.4f} | {hit['path']}"
            )
    else:
        for i, hit in enumerate(hits, 1):
            series = hit["series"] or "—"
            print(
                f"#{i} score={hit['score']:.4f} | {hit['title']} | "
                f"series={series} | chunk={hit['chunk_index']} | {hit['path']}"
            )
            print(hit["content"])
            print("-" * 60)

    conn.close()


if __name__ == "__main__":
    main()
