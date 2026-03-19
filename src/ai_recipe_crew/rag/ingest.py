"""
Data ingestion pipeline: loads recipes, chunks them, and stores in ChromaDB.
"""
import json
import logging
import os
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from ai_recipe_crew.rag.chunker import chunk_all_recipes

logger = logging.getLogger(__name__)

COLLECTION_NAME = "recipes"


def get_chroma_client() -> chromadb.PersistentClient:
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./db")
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=persist_dir)


def get_embedding_function() -> embedding_functions.SentenceTransformerEmbeddingFunction:
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )


def ingest_recipes(force_reingest: bool = False) -> chromadb.Collection:
    """
    Load recipes from JSON, chunk them, and store in ChromaDB.
    Skips ingestion if collection already has data (unless force_reingest=True).
    """
    client = get_chroma_client()
    ef = get_embedding_function()

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() > 0 and not force_reingest:
        logger.info(f"Collection '{COLLECTION_NAME}' already has {collection.count()} documents. Skipping ingestion.")
        return collection

    recipes_path = os.getenv("RECIPES_PATH", "./knowledge/recipes.json")
    if not Path(recipes_path).exists():
        raise FileNotFoundError(f"Recipes file not found at: {recipes_path}")

    with open(recipes_path, "r", encoding="utf-8") as f:
        recipes = json.load(f)

    logger.info(f"Loaded {len(recipes)} recipes from {recipes_path}")

    chunks = chunk_all_recipes(recipes)

    # Clear existing data if force reingest
    if force_reingest and collection.count() > 0:
        client.delete_collection(COLLECTION_NAME)
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )

    documents = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    # Batch upsert in groups of 100
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        collection.upsert(
            documents=documents[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
            ids=ids[i:i + batch_size],
        )

    logger.info(f"Ingested {len(documents)} chunks into ChromaDB collection '{COLLECTION_NAME}'.")
    return collection
