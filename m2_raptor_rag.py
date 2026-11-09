from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from sklearn.mixture import GaussianMixture

from config import chat_client_kwargs, embedding_client_kwargs
from m2_hybrid_rag import all_chunks, rerank
from m2_simple_rag import load_index

RAPTOR_DIR = "chroma_raptor"
RAPTOR_COLLECTION = "company_policies_raptor"

SYSTEM = """You answer questions about company policy.
Use only the context below. If the context does not contain the answer, reply exactly: Not in the knowledge base.
Cite the company and location from the context header, for example [VelocityX, page 23] or [VelocityX, summary level 1]."""


def cluster_chunks(chunks, embeddings, n_clusters):
    """Group similar chunks. Diagonal covariance fits 1536-d embeddings with only a few dozen chunks."""
    vectors = embeddings.embed_documents([doc.page_content for doc in chunks])
    n_clusters = min(n_clusters, len(chunks))
    labels = GaussianMixture(
        n_components=n_clusters,
        covariance_type="diag",
        random_state=42,
    ).fit_predict(vectors)
    groups = {}
    for doc, label in zip(chunks, labels):
        groups.setdefault(int(label), []).append(doc)
    return groups


def summarize_clusters(clusters, llm, level, company):
    summaries = []
    for cluster_id, docs in clusters.items():
        combined = "\n\n".join(doc.page_content for doc in docs)[:8000]
        text = llm.invoke(
            f"Summarize these {company} policy sections. "
            "Keep time windows, product categories, exclusions, and regional rules.\n\n"
            f"{combined}"
        ).content
        summaries.append(Document(
            page_content=text,
            metadata={
                "company": company,
                "doc_type": "policy",
                "level": level,
                "node_type": "summary",
                "children": len(docs),
            },
        ))
        print(f"  {company} level {level}: cluster {cluster_id} has {len(docs)} chunks")
    return summaries


def build_tree(leaves, embeddings, llm):
    for doc in leaves:
        doc.metadata["level"] = 0
        doc.metadata["node_type"] = "leaf"
    nodes = list(leaves)
    for company in sorted({doc.metadata.get("company") for doc in leaves}):
        current = [doc for doc in leaves if doc.metadata.get("company") == company]
        for level in (1, 2):
            if len(current) < 6:
                break
            n_clusters = max(3, len(current) // 4)
            clusters = cluster_chunks(current, embeddings, n_clusters)
            summaries = summarize_clusters(clusters, llm, level, company)
            nodes.extend(summaries)
            current = summaries
    return nodes


def load_or_build():
    embeddings = OpenAIEmbeddings(**embedding_client_kwargs())
    if __import__("pathlib").Path(RAPTOR_DIR).exists():
        print("Loading existing Raptor index")
        return Chroma(
            collection_name=RAPTOR_COLLECTION,
            embedding_function=embeddings,
            persist_directory=RAPTOR_DIR,
        )
    llm = ChatOpenAI(**chat_client_kwargs())
    nodes = build_tree(all_chunks(load_index()), embeddings, llm)
    print(f"Indexing {len(nodes)} nodes")
    return Chroma.from_documents(
        documents=nodes,
        embedding=embeddings,
        collection_name=RAPTOR_COLLECTION,
        persist_directory=RAPTOR_DIR,
    )


def search_level(vector_db, question, node_type):
    return vector_db.similarity_search(
        question,
        k=4,
        filter={"$and": [{"company": "VelocityX"}, {"node_type": node_type}]},
    )


def ask(vector_db, llm, question):
    print(f"\n=== {question} ===")
    candidates = [{"doc": doc} for doc in search_level(vector_db, question, "leaf")]
    candidates += [{"doc": doc} for doc in search_level(vector_db, question, "summary")]
    chosen = rerank(question, candidates, top_n=3)

    blocks = []
    for i, item in enumerate(chosen, start=1):
        doc = item["doc"]
        meta = doc.metadata
        page = meta.get("page")
        where = f"page {page + 1}" if isinstance(page, int) else f"summary level {meta.get('level')}"
        print(
            f"[{i}] level {meta.get('level')} | {meta.get('node_type')} | "
            f"{where} | rerank {item['rerank']:.3f}"
        )
        blocks.append(f"[{i}] VelocityX | {where}\n{doc.page_content}")

    response = llm.invoke([
        SystemMessage(content=SYSTEM),
        HumanMessage(content="Context:\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"),
    ])
    print(response.content)


if __name__ == "__main__":
    vector_db = load_or_build()
    llm = ChatOpenAI(**chat_client_kwargs())
    ask(vector_db, llm, "My football boots tore on concrete. Does VelocityX cover that?")
    ask(vector_db, llm, "Give an overview of VelocityX return windows, exchanges, and regional rules.")