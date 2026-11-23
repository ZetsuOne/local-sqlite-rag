import os
from collections import defaultdict
from pathlib import Path
from mcp.server.mcpserver import MCPServer
from openai import OpenAI
from db import connect
from search import embed_query, search_vec, search_fts

mcp = MCPServer("local-sqlite-rag")

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = os.getenv("RAG_DB_PATH", str(BASE_DIR / "demo.db"))
BASE_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-bge-m3")

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=BASE_URL, api_key="lm-studio")
    return _client


def _rrf_merge(vec_rows, fts_rows, k: int = 60):
    scores = defaultdict(float)
    meta = {}
    for rank, row in enumerate(vec_rows, 1):
        cid = row["chunk_id"]
        scores[cid] += 1.0 / (k + rank)
        meta[cid] = dict(row)
    for rank, row in enumerate(fts_rows, 1):
        cid = row["chunk_id"]
        scores[cid] += 1.0 / (k + rank)
        if cid not in meta:
            meta[cid] = dict(row)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{**meta[cid], "score": score} for cid, score in ranked]


@mcp.tool()
def search_knowledge_base(query: str, series: str | None = None, top_k: int = 10) -> str:
    """Поиск релевантных фрагментов по локальной базе знаний (гибридный векторный + полнотекстовый поиск).
    Используй этот инструмент для поиска фактов, цитат, имён и деталей в проиндексированных документах.
    """
    if not Path(DB_PATH).exists():
        return f"Ошибка: файл базы данных не найден по пути {DB_PATH}"

    conn = connect(DB_PATH)
    try:
        q_vec = embed_query(_get_client(), EMBED_MODEL, query)
        vec_rows = search_vec(conn, q_vec, series, limit=20)
        fts_rows = search_fts(conn, query, series, limit=20)
        merged = _rrf_merge(vec_rows, fts_rows)[:top_k]
    finally:
        conn.close()

    if not merged:
        return "По данному запросу в базе знаний ничего не найдено."

    blocks = []
    for i, item in enumerate(merged, 1):
        blocks.append(
            f"[{i}] {item['title']} (series: {item['series']}, chunk: {item['chunk_index']}, score: {item['score']:.4f})\n"
            f"{item['content']}"
        )
    return "\n\n---\n\n".join(blocks)


@mcp.tool()
def list_series() -> str:
    """Возвращает список доступных циклов/серий (series) и количество проиндексированных книг в базе."""
    if not Path(DB_PATH).exists():
        return f"Ошибка: файл базы данных не найден по пути {DB_PATH}"

    conn = connect(DB_PATH)
    try:
        rows = conn.execute(
            "SELECT series, COUNT(*) FROM files GROUP BY series ORDER BY series"
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return "База знаний пуста."
    return "\n".join(f"- {s}: {cnt} файл(ов)" for s, cnt in rows)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
