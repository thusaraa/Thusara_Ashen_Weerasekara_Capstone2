from langchain_community.retrievers import BM25Retriever
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import chat_client_kwargs
from m2_hybrid_rag import CANDIDATES, RRF_K, SYSTEM, all_chunks, page_label, rerank, tokenize
from m2_simple_rag import load_index

from m2_hybrid_rag import get_reranker


def rewrite_queries(llm, question):
    """Three other ways a customer might ask the same thing."""
    response = llm.invoke(
        "Write 3 search queries a written policy would match. "
        "Use policy wording: US means North America, and a free shipping label means a prepaid return label. "
        "Keep the company name. One query per line, with no quotes.\n\n"
        f"{question}"
)
    queries = []
    for line in response.content.splitlines():
        cleaned = line.strip().strip('"').lstrip("0123456789.-) ").strip().strip('"')
        if cleaned:
            queries.append(cleaned)
    return queries[:3]


def rrf_fuse(ranked_lists, top_n=CANDIDATES):
    """Same 1/(60+rank) score, added across every query's semantic list and keyword list."""
    scores = {}
    for docs in ranked_lists:
        for rank, doc in enumerate(docs, start=1):
            entry = scores.setdefault(doc.page_content, {"doc": doc, "score": 0.0})
            entry["score"] += 1.0 / (RRF_K + rank)
    ordered = sorted(scores.values(), key=lambda item: item["score"], reverse=True)
    return ordered[:top_n]

def best_rerank(queries, fused, top_n=5):
    """Score every chunk against every query and keep the highest score."""
    docs = [item["doc"] for item in fused]
    best = [float("-inf")] * len(docs)
    for query in queries:
        scores = get_reranker().predict([(query, doc.page_content) for doc in docs])
        for i, score in enumerate(scores):
            best[i] = max(best[i], float(score))
    for item, score in zip(fused, best):
        item["rerank"] = score
    ordered = sorted(fused, key=lambda item: item["rerank"], reverse=True)
    return ordered[:top_n]

def ask(vector_db, keyword_retriever, llm, question):
    print(f"\n=== {question} ===")
    queries = [question, *rewrite_queries(llm, question)]
    for query in queries:
        print(" -", query)

    ranked = []
    for query in queries:
        ranked.append(vector_db.similarity_search(query, k=20))
        ranked.append(keyword_retriever.invoke(query)[:20])
    fused = rrf_fuse(ranked)
    seen = {item["doc"].page_content for item in fused}
    for docs in ranked:
        if docs and docs[0].page_content not in seen:
            fused.append({"doc": docs[0], "score": 0.0})
            seen.add(docs[0].page_content)
    fused = best_rerank(queries, fused)

    blocks = []
    for i, item in enumerate(fused, start=1):
        doc = item["doc"]
        meta = doc.metadata
        print(
            f"[{i}] {meta.get('company')} | page {page_label(meta)} | "
            f"rerank {item['rerank']:.3f} | rrf {item['score']:.4f}"
        )
        blocks.append(f"[{i}] {meta.get('company')} | page {page_label(meta)}\n{doc.page_content}")

    response = llm.invoke([
        SystemMessage(content=SYSTEM),
        HumanMessage(content="Context:\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"),
    ])
    print(response.content)


if __name__ == "__main__":
    vector_db = load_index()
    keyword_retriever = BM25Retriever.from_documents(all_chunks(vector_db), preprocess_func=tokenize)
    keyword_retriever.k = 50
    llm = ChatOpenAI(**chat_client_kwargs())
    ask(
        vector_db,
        keyword_retriever,
        llm,
        "Can I get a free shipping label if I send back VelocityX shoes I bought in the US?",
    )