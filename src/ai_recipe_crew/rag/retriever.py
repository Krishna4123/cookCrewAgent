"""
Vector search retriever for recipe chunks.
"""
import logging
import os
from typing import Any

import chromadb

from ai_recipe_crew.rag.ingest import get_chroma_client, get_embedding_function, COLLECTION_NAME

logger = logging.getLogger(__name__)


class RecipeRetriever:
    """Handles semantic search over the ChromaDB recipe collection."""

    def __init__(self):
        self._client: chromadb.PersistentClient | None = None
        self._collection: chromadb.Collection | None = None

    def _get_collection(self) -> chromadb.Collection:
        if self._collection is None:
            self._client = get_chroma_client()
            ef = get_embedding_function()
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=ef,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        chunk_type_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve top-k relevant recipe chunks for a given query.

        Args:
            query: Natural language search query.
            top_k: Number of results to return. Defaults to TOP_K_RESULTS env var.
            chunk_type_filter: Optional filter by chunk type (overview, ingredients, steps, nutrition).

        Returns:
            List of dicts with 'text', 'metadata', and 'distance'.
        """
        if top_k is None:
            top_k = int(os.getenv("TOP_K_RESULTS", "5"))

        collection = self._get_collection()

        if collection.count() == 0:
            logger.warning("ChromaDB collection is empty. Run ingestion first.")
            return []

        where_filter = {"chunk_type": chunk_type_filter} if chunk_type_filter else None

        try:
            results = collection.query(
                query_texts=[query],
                n_results=min(top_k, collection.count()),
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error(f"ChromaDB query failed: {e}")
            return []

        output = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            output.append({
                "text": doc,
                "metadata": meta,
                "distance": dist,
            })

        logger.info(f"Retrieved {len(output)} chunks for query: '{query[:60]}...'")
        return output

    def retrieve_formatted(self, query: str, top_k: int | None = None) -> str:
        """Returns retrieved chunks as a formatted string for LLM context."""
        chunks = self.retrieve(query, top_k=top_k)
        if not chunks:
            return "No relevant recipes found in the knowledge base."

        lines = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk["metadata"]
            lines.append(
                f"[Result {i}] Recipe: {meta.get('recipe_name', 'Unknown')} "
                f"(Type: {meta.get('chunk_type', 'N/A')}, "
                f"Similarity: {1 - chunk['distance']:.2f})\n"
                f"{chunk['text']}\n"
            )
        return "\n".join(lines)


# Singleton instance
_retriever_instance: RecipeRetriever | None = None


def get_retriever() -> RecipeRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = RecipeRetriever()
    return _retriever_instance
