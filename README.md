# Capstone 2 — Company Knowledge-Base Chatbot

Answers questions about the VelocityX and LuxeThreads policy PDFs. Retrieval is hybrid BM25 plus semantic search, fused with reciprocal rank fusion and reranked by a cross-encoder. Chat and embeddings go through OpenRouter.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Put the two policy PDFs in `data/`. 

## Run

Run these in order. Later scripts read the Chroma index created by ingestion.

```powershell
.\.venv\Scripts\python.exe m2_chroma_ingestion.py
.\.venv\Scripts\python.exe m2_simple_rag.py
.\.venv\Scripts\python.exe m2_hybrid_rag.py
.\.venv\Scripts\python.exe m2_metadata_filtering.py
.\.venv\Scripts\python.exe m2_raptor_rag.py
.\.venv\Scripts\python.exe evaluate.py
.\.venv\Scripts\python.exe trace_one.py
.\.venv\Scripts\python.exe m2_multi_query_rag.py
```

`evaluate.py` writes `evaluation_report.md`. `trace_one.py` prints LangSmith trace URLs when tracing is on.

Ingestion deletes `chroma_db` and rebuilds it, so a second run does not duplicate chunks. Raptor keeps `chroma_raptor` once that folder exists.

## What each file does

- `config.py` — paths, chunk size, corpus metadata, and the OpenRouter client settings.
- `m2_chroma_ingestion.py` — loads both PDFs, stamps company metadata, splits the text, and stores chunks in the `company_policies` collection.
- `m2_simple_rag.py` — semantic search only. Also exposes `load_index()` for the other scripts.
- `m2_hybrid_rag.py` — BM25 plus semantic search, fused by reciprocal rank fusion, then reranked with `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- `m2_metadata_filtering.py` — the same hybrid path with a company filter on both the vector search and the keyword search.
- `m2_raptor_rag.py` — clusters chunks per company, stores leaf and summary nodes in `chroma_raptor`, and searches each level separately.
- `m2_multi_query_rag.py` — rewrites the question into policy wording, fuses every query, and keeps the best cross-encoder score.
- `evaluate.py` — precision@5, recall@5, and latency for semantic search against hybrid search plus reranking.
- `data/questions.json` — the labeled questions used by the evaluation.
- `trace_one.py` — one semantic retrieval and one hybrid retrieval, sent to the `capstone-2` LangSmith project.

## Flowise

`query_rerank_redismem.json` and `flowise/Capstone 2 Chatflow.json` are the same chatflow. It uses multi-query retrieval, the local Chroma collection, and buffer memory. The chat model and embeddings use base path `https://openrouter.ai/api/v1`. Paste the OpenRouter key into the Flowise OpenAI credential. The model names are `openai/gpt-4o-mini` and `openai/text-embedding-3-small`.

Start Chroma before opening the chat, and leave that process running:

```powershell
.\.venv\Scripts\chroma.exe run --path ".\chroma_db" --host localhost --port 8000
```

Then start Flowise with `npx flowise start` and open `http://localhost:3000`. The Chroma node URL is `http://localhost:8000` and the collection is `company_policies`.

While that server is running it locks `chroma_db`, so the Python scripts that open the same folder will fail until you stop it.
