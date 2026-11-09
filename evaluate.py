import json
import time
from pathlib import Path

from langchain_community.retrievers import BM25Retriever

from m2_hybrid_rag import CANDIDATES, all_chunks, page_label, rerank, rrf_merge, tokenize, get_reranker
from m2_simple_rag import load_index

QUESTIONS = Path("data/questions.json")
REPORT = Path("evaluation_report.md")


def labels(docs):
    return [(doc.metadata.get("company"), page_label(doc.metadata)) for doc in docs]


def score(found, company, pages):
    relevant = [item for item in found if item == (company, pages[0]) or (item[0] == company and item[1] in pages)]
    precision = len(relevant) / len(found) if found else 0.0
    found_pages = {page for found_company, page in found if found_company == company and page in pages}
    recall = len(found_pages) / len(pages)
    return precision, recall


def semantic_retrieve(vector_db, question):
    return vector_db.similarity_search(question, k=5)


def hybrid_retrieve(vector_db, keyword_retriever, question):
    semantic_docs = vector_db.similarity_search(question, k=50)
    keyword_docs = keyword_retriever.invoke(question)
    fused = rerank(question, rrf_merge(semantic_docs, keyword_docs, top_n=CANDIDATES))
    return [item["doc"] for item in fused]


def timed(fn):
    start = time.perf_counter()
    docs = fn()
    elapsed_ms = (time.perf_counter() - start) * 1000
    return docs, elapsed_ms


def main():
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    vector_db = load_index()
    chunks = all_chunks(vector_db)
    keyword_retriever = BM25Retriever.from_documents(chunks, preprocess_func=tokenize)
    keyword_retriever.k = 50
    get_reranker()

    rows = []
    for item in questions:
        semantic_docs, semantic_ms = timed(lambda q=item["question"]: semantic_retrieve(vector_db, q))
        hybrid_docs, hybrid_ms = timed(
            lambda q=item["question"]: hybrid_retrieve(vector_db, keyword_retriever, q)
        )
        for mode, docs, elapsed in (
            ("semantic", semantic_docs, semantic_ms),
            ("hybrid+rerank", hybrid_docs, hybrid_ms),
        ):
            precision, recall = score(labels(docs), item["company"], item["pages"])
            rows.append({
                "question": item["question"],
                "mode": mode,
                "precision": precision,
                "recall": recall,
                "latency_ms": elapsed,
            })
            print(f"{mode:16} P {precision:.2f} R {recall:.2f} {elapsed:7.0f} ms | {item['question']}")

    lines = [
        "# Retrieval evaluation",
        "",
        "Precision@5 is the fraction of the five retrieved chunks whose company and page match the label. Recall@5 is 1 when every labeled page appears at least once.",
        "",
        "| Question | Mode | Precision@5 | Recall@5 | Latency ms |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['question']} | {row['mode']} | {row['precision']:.2f} | "
            f"{row['recall']:.2f} | {row['latency_ms']:.0f} |"
        )
    lines += ["", "## Averages", ""]
    for mode in ("semantic", "hybrid+rerank"):
        subset = [row for row in rows if row["mode"] == mode]
        lines.append(
            f"- {mode}: precision {sum(r['precision'] for r in subset) / len(subset):.2f}, "
            f"recall {sum(r['recall'] for r in subset) / len(subset):.2f}, "
            f"latency {sum(r['latency_ms'] for r in subset) / len(subset):.0f} ms"
        )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {REPORT}")


if __name__ == "__main__":
    main()