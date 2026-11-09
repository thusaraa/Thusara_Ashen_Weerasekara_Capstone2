import shutil

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    CORPUS,
    DATA_DIR,
    embedding_client_kwargs,
)


def load_corpus():
    """Load both policies and stamp file-level metadata onto each page."""
    pages = []
    for filename, meta in CORPUS.items():
        path = DATA_DIR / filename
        if not path.exists():
            raise FileNotFoundError(path)

        documents = PyPDFLoader(str(path)).load()
        for doc in documents:
            doc.metadata.update(meta)
        print(f"{meta['company']}: {len(documents)} pages from {filename}")
        pages.extend(documents)
    return pages


def split_pages(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(pages)


def print_chunk_diagnostics(chunks, max_chunks=3):
    print(f"\nTotal chunks: {len(chunks)}")
    for company in ("VelocityX", "LuxeThreads"):
        count = sum(1 for c in chunks if c.metadata.get("company") == company)
        print(f"  {company}: {count}")

    print(f"\nFirst {min(max_chunks, len(chunks))} chunks:\n")
    for i, chunk in enumerate(chunks[:max_chunks], start=1):
        meta = chunk.metadata
        print(f"--- Chunk {i} ---")
        print(
            f"{meta.get('company')} | {meta.get('doc_type')} | "
            f"page {meta.get('page')} | {meta.get('source')}"
        )
        print(chunk.page_content[:400].replace("\n", " "))
        print()


def ingest():
    # Rebuild from scratch so a second run does not duplicate chunks.
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    pages = load_corpus()
    chunks = split_pages(pages)
    print_chunk_diagnostics(chunks)

    embeddings = OpenAIEmbeddings(**embedding_client_kwargs())
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )
    print(f"Stored {len(chunks)} chunks in {CHROMA_DIR}")


if __name__ == "__main__":
    ingest()