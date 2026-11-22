import argparse
import json
import re
from openai import OpenAI
from db import connect
from search import embed_query, search_vec, search_fts, rrf_fuse, answer_with_llm
JUDGE_SYSTEM = (
    "Ты строгий оценщик качества RAG-системы. Оцени ответ модели по двум критериям от 0.0 до 1.0:\n"
    "1. faithfulness: 1.0 — все факты в ответе строго подтверждены контекстом; "
    "снижай оценку за любые догадки, искажения или факты извне контекста.\n"
    "2. answer_relevance: 1.0 — ответ прямо, полно и по существу отвечает на вопрос пользователя.\n"
    'Верни ТОЛЬКО валидный JSON формата: {"faithfulness": 0.0, "answer_relevance": 0.0, "reason": "кратко"}'
)
def parse_json_response(text: str) -> dict:
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", clean, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {"faithfulness": 0.0, "answer_relevance": 0.0, "reason": "JSON parse error"}
def evaluate_retrieval(hits: list[dict], expected_substrs: list[str], expected_keywords: list[str]) -> tuple[float, float, float]:
    if not expected_substrs:
        return 1.0, 1.0, 1.0
    substrs_low = [s.lower() for s in expected_substrs]
    hit_rank = None
    for idx, row in enumerate(hits, start=1):
        title = (row.get("title") or "").lower()
        if any(sub in title for sub in substrs_low):
            hit_rank = idx
            break
    recall = 1.0 if hit_rank is not None else 0.0
    mrr = (1.0 / hit_rank) if hit_rank is not None else 0.0
    combined_text = "\n".join((r.get("content") or "").lower() for r in hits)
    if expected_keywords:
        matched = sum(1 for kw in expected_keywords if kw.lower() in combined_text)
        kw_rate = matched / len(expected_keywords)
    else:
        kw_rate = 1.0
    return recall, mrr, kw_rate
def judge_answer(client: OpenAI, model: str, query: str, hits: list[dict], answer: str) -> dict:
    context = "\n\n---\n\n".join(f"[{r.get('title', '')}]\n{r.get('content', '')}" for r in hits)
    prompt = f"Вопрос: {query}\n\nКонтекст:\n{context}\n\nОтвет модели:\n{answer}"
    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    return parse_json_response(resp.choices[0].message.content or "")
def main():
    parser = argparse.ArgumentParser(description="Evaluate local-sqlite-rag")
    parser.add_argument("--db", required=True, help="Path to SQLite DB")
    parser.add_argument("--eval-file", required=True, help="Path to *_eval.json")
    parser.add_argument("--top-k", type=int, default=10, help="Top K chunks to evaluate")
    parser.add_argument("--url", default="http://localhost:1234/v1", help="LM Studio URL")
    parser.add_argument("--embed-model", default="text-embedding-bge-m3", help="Embedding model")
    parser.add_argument("--chat-model", default=None, help="Chat/Judge model (optional)")
    args = parser.parse_args()
    with open(args.eval_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    conn = connect(args.db)
    client = OpenAI(base_url=args.url, api_key="lm-studio")
    recalls, mrrs, kw_rates = [], [], []
    faiths, relevances = [], []
    for i, item in enumerate(dataset, start=1):
        query = item["query"]
        series = item.get("series")
        exp_titles = item.get("expected_title_substr", [])
        exp_kws = item.get("expected_keywords", [])
        q_vec = embed_query(client, args.embed_model, query)
        limit = max(20, args.top_k * 2)
        vec_rows = search_vec(conn, q_vec, series, limit=limit)
        fts_rows = search_fts(conn, query, series, limit=limit)
        hits = rrf_fuse(vec_rows, fts_rows, top_k=args.top_k)

        rec, mrr, kw = evaluate_retrieval(hits, exp_titles, exp_kws)
        recalls.append(rec)
        mrrs.append(mrr)
        kw_rates.append(kw)
        status = f"[{i}/{len(dataset)}] Recall@{args.top_k}: {rec:.0%} | MRR: {mrr:.2f} | KW: {kw:.0%}"
        if args.chat_model:
            ans = answer_with_llm(client, args.chat_model, query, hits)
            scores = judge_answer(client, args.chat_model, query, hits, ans)
            f_score = float(scores.get("faithfulness", 0.0))
            r_score = float(scores.get("answer_relevance", 0.0))
            faiths.append(f_score)
            relevances.append(r_score)
            status += f" | Faith: {f_score:.2f} | Rel: {r_score:.2f} ({scores.get('reason', '')})"
        print(status)
        print(f"   Q: {query}")
    n = len(dataset)
    print("\n" + "=" * 50)
    print(f"ИТОГО ({n} вопросов, top_k={args.top_k}):")
    print(f"Recall@{args.top_k}:      {sum(recalls) / n:.1%}")
    print(f"MRR:           {sum(mrrs) / n:.3f}")
    print(f"Keyword Hit:   {sum(kw_rates) / n:.1%}")
    if args.chat_model and faiths:
        print(f"Faithfulness:  {sum(faiths) / n:.3f}")
        print(f"Relevance:     {sum(relevances) / n:.3f}")
if __name__ == "__main__":
    main()
