from dotenv import load_dotenv

load_dotenv()

from langchain_community.retrievers import BM25Retriever
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from config import chat_client_kwargs
from evaluate import hybrid_retrieve, semantic_retrieve
from m2_hybrid_rag import SYSTEM, all_chunks, page_label, tokenize
from m2_simple_rag import load_index


def hits(docs):
    return [
        {"company": doc.metadata.get("company"), "page": page_label(doc.metadata)}
        for doc in docs
    ]


@traceable(run_type="retriever", name="semantic_search")
def traced_semantic(vector_db, question):
    return hits(semantic_retrieve(vector_db, question))


@traceable(run_type="retriever", name="hybrid_rerank")
def traced_hybrid(vector_db, keyword_retriever, question):
    docs = hybrid_retrieve(vector_db, keyword_retriever, question)
    blocks = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        blocks.append(f"[{i}] {meta.get('company')} | page {page_label(meta)}\n{doc.page_content}")
    return {"hits": hits(docs), "context": "\n\n".join(blocks)}


@traceable(run_type="chain", name="policy_question")
def ask(vector_db, keyword_retriever, llm, question):
    print(f"\n=== {question} ===")
    print("semantic", traced_semantic(vector_db, question))
    hybrid = traced_hybrid(vector_db, keyword_retriever, question)
    print("hybrid", hybrid["hits"])
    response = llm.invoke([
        SystemMessage(content=SYSTEM),
        HumanMessage(content=f"Context:\n{hybrid['context']}\n\nQuestion: {question}"),
    ])
    print(response.content)
    run = get_current_run_tree()
    if run is not None:
        print(run.get_url())
    return response.content


if __name__ == "__main__":
    vector_db = load_index()
    keyword_retriever = BM25Retriever.from_documents(all_chunks(vector_db), preprocess_func=tokenize)
    keyword_retriever.k = 50
    llm = ChatOpenAI(**chat_client_kwargs())
    ask(vector_db, keyword_retriever, llm, "Does VelocityX offer prepaid return labels in North America?")
    ask(vector_db, keyword_retriever, llm, "How long is the LuxeThreads return window for fine jewelry?")