import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "company_policies"

# The query step must use this same model.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Policy text. Large enough to keep a rule intact, with overlap so a
# rule cut at the edge still appears in the next chunk.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Metadata copied onto every chunk from that file.
# source and page are added by the PDF loader, not here.
CORPUS = {
    "VelocityX Ecommerce Warranty, Returns & Exchange Policy.pdf": {
        "company": "VelocityX",
        "doc_type": "policy",
        "department": "support",
        "version": "1.0",
        "date": "",
    },
    "Warranty_Returns_Policy_LuxeThreads.pdf": {
        "company": "LuxeThreads",
        "doc_type": "policy",
        "department": "support",
        "version": "3.2",
        "date": "2025-01-01",
    },
}


def embedding_client_kwargs() -> dict:
    """Arguments for OpenAIEmbeddings. OpenAI key wins when both are set."""
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        return {"model": EMBEDDING_MODEL, "api_key": openai_key}

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        model = EMBEDDING_MODEL
        if "/" not in model:
            model = f"openai/{model}"
        return {
            "model": model,
            "api_key": openrouter_key,
            "base_url": "https://openrouter.ai/api/v1",
        }

    raise RuntimeError("Set OPENAI_API_KEY or OPENROUTER_API_KEY in .env")

def chat_client_kwargs() -> dict:
    """Arguments for ChatOpenAI. Uses the same key as the embeddings."""
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        model = os.getenv("CHAT_MODEL", "gpt-4o-mini")
        return {"model": model, "api_key": openai_key, "temperature": 0}

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        model = os.getenv("CHAT_MODEL", "openai/gpt-4o-mini")
        return {
            "model": model,
            "api_key": openrouter_key,
            "base_url": "https://openrouter.ai/api/v1",
            "temperature": 0,
        }

    raise RuntimeError("Set OPENAI_API_KEY or OPENROUTER_API_KEY in .env")