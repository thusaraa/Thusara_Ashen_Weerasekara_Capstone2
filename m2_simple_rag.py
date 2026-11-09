from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    chat_client_kwargs,
    embedding_client_kwargs,
)

SYSTEM = """You answer questions about company policy.
Use only the context below. If the context does not contain the answer, reply exactly: Not in the knowledge base.
Cite each claim as [Company, page N], using the page number from the context header."""


def load_index():
    embeddings = OpenAIEmbeddings(**embedding_client_kwargs())
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def format_context(docs_and_scores):
    """Turn retrieved chunks into a numbered block the model can cite."""
    blocks = []
    for i, (doc, score) in enumerate(docs_and_scores, start=1):
        meta = doc.metadata
        page = meta.get("page")
        page_label = page + 1 if isinstance(page, int) else page
        header = (
            f"[{i}] {meta.get('company')} | page {page_label} | "
            f"distance {score:.3f}"
        )
        blocks.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(blocks)


def ask(vector_db, llm, question, k=4):
    print(f"\n=== {question} ===")
    # Distance: a smaller number is a closer chunk.
    docs_and_scores = vector_db.similarity_search_with_score(question, k=k)
    context = format_context(docs_and_scores)
    print(context[:1200])
    print()

    response = llm.invoke([
        SystemMessage(content=SYSTEM),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
    ])
    print(response.content)
    return response.content


if __name__ == "__main__":
    vector_db = load_index()
    llm = ChatOpenAI(**chat_client_kwargs())

    ask(vector_db, llm, "My football boots tore on concrete. Does VelocityX cover that?")
    ask(vector_db, llm, "How long is the LuxeThreads return window for fine jewelry?")
    ask(vector_db, llm, "What is the warranty on a Tesla battery?")