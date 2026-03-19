"""
CrewAI tool wrapping the RAG retriever for agent use.
"""
import logging
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from ai_recipe_crew.rag.retriever import get_retriever

logger = logging.getLogger(__name__)


class RecipeSearchInput(BaseModel):
    query: str = Field(description="Natural language query to search for relevant recipes.")
    top_k: int = Field(default=5, description="Number of top results to retrieve.")


class RecipeRetrieverTool(BaseTool):
    name: str = "Recipe Knowledge Base Search"
    description: str = (
        "Search the recipe knowledge base using semantic similarity. "
        "Use this tool to find relevant recipes based on ingredients, cuisine type, "
        "flavor preferences, or cooking methods. Returns the most relevant recipe chunks."
    )
    args_schema: type[BaseModel] = RecipeSearchInput

    def _run(self, query: str, top_k: int = 5) -> str:
        try:
            retriever = get_retriever()
            result = retriever.retrieve_formatted(query=query, top_k=top_k)
            return result
        except Exception as e:
            logger.error(f"RecipeRetrieverTool failed: {e}")
            return f"Error retrieving recipes: {str(e)}"
