from langchain_community.retrievers import BM25Retriever
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import chat_client_kwargs
from m2_hybrid_rag import (
    CANDIDATES,
    SYSTEM,
    all_chunks,
    page_label,
    rerank,
    rrf_merge,
    tokenize,
)
from m2_simple_rag import load_index


def keyword_search(chunks, question, metadata_filter):
    """Rank only chunks whose metadata matches every filter field."""
    pool = [
        doc for doc in chunks
        if all(doc.metadata.get(key) == value for key, value in metadata_filter.items())
    ]
    retriever = BM25Retriever.from_documents(pool, preprocess_func=tokenize)
    retriever.k = min(50, len(pool))
    return retriever.invoke(question)


def ask(vector_db, chunks, llm, question, metadata_filter=None):
    metadata_filter = metadata_filter or {}
    label = metadata_filter or "no filter"
    print(f"\n=== {question} | {label} ===")

    semantic_docs = vector_db.similarity_search(question, k=50, filter=metadata_filter or None)
    keyword_docs = keyword_search(chunks, question, metadata_filter)
    fused = rerank(question, rrf_merge(semantic_docs, keyword_docs, top_n=CANDIDATES))

    blocks = []
    for i, item in enumerate(fused, start=1):
        doc = item["doc"]
        meta = doc.metadata
        print(
            f"[{i}] {meta.get('company')} | page {page_label(meta)} | "
            f"rerank {item['rerank']:.3f}"
        )
        blocks.append(
            f"[{i}] {meta.get('company')} | page {page_label(meta)}\n{doc.page_content}"
        )

    response = llm.invoke([
        SystemMessage(content=SYSTEM),
        HumanMessage(content="Context:\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"),
    ])
    print(response.content)


if __name__ == "__main__":
    vector_db = load_index()
    chunks = all_chunks(vector_db)
    llm = ChatOpenAI(**chat_client_kwargs())

    jewelry = "How long is the LuxeThreads return window for fine jewelry?"
    ask(vector_db, chunks, llm, jewelry)
    ask(vector_db, chunks, llm, jewelry, {"company": "VelocityX"})

    footwear = "What is the standard return window for footwear?"
    ask(vector_db, chunks, llm, footwear, {"company": "VelocityX"})
    ask(vector_db, chunks, llm, footwear, {"company": "LuxeThreads"})