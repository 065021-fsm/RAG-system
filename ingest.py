"""
ingest.py — Load dataset, chunk text, embed via Ollama, store in PostgreSQL/pgvector.
"""

import sys
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from langchain_core.documents import Document
from config import load_config


def load_dataset(dataset_path: str) -> list[Document]:
    """Load dataset from a .txt file and return as LangChain Documents."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    # Split the content into paragraphs/sections
    # The dataset has numbered sections separated by newlines
    sections = []
    current_section = []

    for line in content.split("\n"):
        line = line.strip()
        if not line:
            if current_section:
                sections.append(" ".join(current_section))
                current_section = []
        else:
            current_section.append(line)

    if current_section:
        sections.append(" ".join(current_section))

    documents = []
    for i, section in enumerate(sections):
        if section.strip():
            documents.append(
                Document(
                    page_content=section.strip(),
                    metadata={"source": str(dataset_path), "section_index": i},
                )
            )

    print(f"[Ingest] Loaded {len(documents)} sections from {dataset_path}")
    return documents


def chunk_documents(documents: list[Document], chunk_size: int = 500, chunk_overlap: int = 100) -> list[Document]:
    """Split documents into smaller chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"[Ingest] Split into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks


def create_vector_store(chunks: list[Document], config) -> PGVector:
    """Create embeddings and store in PostgreSQL/pgvector."""
    embeddings = OllamaEmbeddings(
        model=config.embedding_model_name,
        base_url=config.ollama_base_url,
    )

    print(f"[Ingest] Creating vector store with {len(chunks)} chunks...")
    print(f"[Ingest] Embedding model: {config.embedding_model_name}")
    print(f"[Ingest] Database: {config.db_name} @ {config.db_host}")

    vector_store = PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=config.collection_name,
        connection=config.pg_connection_string,
        pre_delete_collection=True,  # Clean slate each run
    )

    print(f"[Ingest] Vector store created successfully!")
    return vector_store


def run_ingestion(config=None):
    """Run the full ingestion pipeline."""
    if config is None:
        config = load_config()

    # Load
    documents = load_dataset(config.dataset_path)

    # Chunk
    chunks = chunk_documents(documents)

    # Embed & Store
    vector_store = create_vector_store(chunks, config)

    return vector_store


if __name__ == "__main__":
    run_ingestion()
