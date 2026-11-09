from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from m2_simple_rag import load_index
from sentence_transformers import CrossEncoder

SYSTEM = """You answer questions about company policy.
Use only the context below. If the context does not contain the answer, reply exactly: Not in the knowledge base.
Cite each claim with the company and page from the context header, for example [VelocityX, page 23].
The number in brackets is the chunk order. Cite the page written after the word page in the header."""

RRF_K = 60
CANDIDATES = 20
TOP_N = 5

_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker

def rerank(question, fused, top_n=TOP_N):
    """Score each question-chunk pair. A higher score is a better match."""
    scores = get_reranker().predict(
        [(question, item["doc"].page_content) for item in fused]
    )
    for item, score in zip(fused, scores):
        item["rerank"] = float(score)
    ordered = sorted(fused, key=lambda item: item["rerank"], reverse=True)
    return ordered[:top_n]


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def all_chunks(vector_db) -> list[Document]:
    raw = vector_db.get(include=["documents", "metadatas"])
    return [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]


def rrf_merge(semantic_docs, keyword_docs, top_n=TOP_N):
    """Fuse two ranked lists. Rank 1 scores 1/61, and a chunk on both lists adds both scores."""
    scores = {}
    for source, docs in (("semantic", semantic_docs), ("keyword", keyword_docs)):
        for rank, doc in enumerate(docs, start=1):
            entry = scores.setdefault(
                doc.page_content,
                {"doc": doc, "score": 0.0, "semantic": None, "keyword": None},
            )
            entry["score"] += 1.0 / (RRF_K + rank)
            entry[source] = rank
    ordered = sorted(scores.values(), key=lambda item: item["score"], reverse=True)
    return ordered[:top_n]


def page_label(meta):
    page = meta.get("page")
    return page + 1 if isinstance(page, int) else page


def ask(vector_db, keyword_retriever, llm, question):
    print(f"\n=== {question} ===")
    semantic_docs = vector_db.similarity_search(question, k=50)
    keyword_docs = keyword_retriever.invoke(question)
    fused = rerank(question, rrf_merge(semantic_docs, keyword_docs, top_n=CANDIDATES))

    blocks = []
    for i, item in enumerate(fused, start=1):
        doc = item["doc"]
        meta = doc.metadata
        print(
            f"[{i}] {meta.get('company')} | page {page_label(meta)} | "
            f"rerank {item['rerank']:.3f} | rrf {item['score']:.4f} | "
            f"semantic rank {item['semantic']} | keyword rank {item['keyword']}"
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
    keyword_retriever = BM25Retriever.from_documents(
        all_chunks(vector_db),
        preprocess_func=tokenize,
    )
    keyword_retriever.k = 50
    llm = ChatOpenAI(**__import__("config").chat_client_kwargs())

    ask(vector_db, keyword_retriever, llm, "My football boots tore on concrete. Does VelocityX cover that?")
    ask(vector_db, keyword_retriever, llm, "How long is the LuxeThreads return window for fine jewelry?")
    ask(vector_db, keyword_retriever, llm, "What is the warranty on a Tesla battery?")