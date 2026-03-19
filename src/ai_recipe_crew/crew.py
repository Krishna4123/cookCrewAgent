"""
CrewAI orchestration with a single-agent design to minimize token usage.
Falls back to direct RAG + LLM call if CrewAI fails.
"""
import json
import logging
import os
from typing import Any

from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM
from groq import Groq

from ai_recipe_crew.rag.retriever import get_retriever
from ai_recipe_crew.tools.retriever_tool import RecipeRetrieverTool
from ai_recipe_crew.utils.parser import safe_parse_recipe

logger = logging.getLogger(__name__)

# Strict output schema injected into every prompt
_OUTPUT_SCHEMA = '''{
  "name": "recipe name",
  "ingredients": ["ingredient 1", "ingredient 2"],
  "steps": ["step 1", "step 2"],
  "time": "cooking time",
  "nutrition": {
    "calories": "value with unit",
    "protein": "value with unit",
    "carbs": "value with unit",
    "fat": "value with unit"
  }
}'''


def _build_llm() -> LLM:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set.")
    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    return LLM(model=f"groq/{model}", api_key=api_key, temperature=0.3)


def _direct_llm_fallback(query: str, inventory: list[str]) -> dict[str, Any]:
    """
    Fallback: retrieve RAG context directly and call Groq API without CrewAI.
    Uses minimal tokens — one single LLM call.
    """
    logger.info("Using direct LLM fallback.")
    retriever = get_retriever()
    context = retriever.retrieve_formatted(query=query, top_k=3)

    inventory_str = ", ".join(inventory) if inventory else "not specified"

    prompt = f"""You are a recipe assistant. Based on the context below, suggest one recipe.

RAG CONTEXT:
{context}

USER QUERY: {query}
AVAILABLE INVENTORY: {inventory_str}

Respond ONLY with a valid JSON object matching this exact schema (no extra text):
{_OUTPUT_SCHEMA}"""

    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=800,
    )
    raw = response.choices[0].message.content
    return safe_parse_recipe(raw)


def run_recipe_crew(query: str, inventory: list[str] | None = None) -> dict[str, Any]:
    """
    Run a single-agent CrewAI pipeline. Falls back to direct LLM call on failure.
    """
    if inventory is None:
        inventory = []

    logger.info(f"Starting recipe crew | query='{query}' | inventory={inventory}")

    inventory_str = ", ".join(inventory) if inventory else "not specified (assume all available)"

    try:
        llm = _build_llm()
        retriever_tool = RecipeRetrieverTool()

        agent = Agent(
            role="Recipe Assistant",
            goal="Find the best matching recipe and return strict JSON output.",
            backstory="You are a culinary expert who finds recipes and outputs structured JSON.",
            tools=[retriever_tool],
            llm=llm,
            verbose=False,
            allow_delegation=False,
            max_iter=3,
        )

        task = Task(
            description=f"""Search for a recipe matching: "{query}"
Available inventory: {inventory_str}

Steps:
1. Use the search tool ONCE to find relevant recipes.
2. Pick the best match considering the query and inventory.
3. Estimate nutrition if not available.
4. Output ONLY this JSON (no extra text):
{_OUTPUT_SCHEMA}""",
            expected_output="A single valid JSON object with keys: name, ingredients, steps, time, nutrition.",
            agent=agent,
        )

        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=False,
        )

        result = crew.kickoff()
        raw = result.raw if hasattr(result, "raw") else str(result)
        logger.info("Crew execution completed.")
        return safe_parse_recipe(raw)

    except Exception as e:
        logger.warning(f"CrewAI failed ({type(e).__name__}: {e}). Switching to direct fallback.")
        return _direct_llm_fallback(query=query, inventory=inventory)
