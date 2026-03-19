"""
CrewAI orchestration: defines agents, tasks, and the crew pipeline.
"""
import logging
import os
from typing import Any

from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ai_recipe_crew.tools.retriever_tool import RecipeRetrieverTool
from ai_recipe_crew.utils.loader import load_agents_config, load_tasks_config
from ai_recipe_crew.utils.parser import safe_parse_recipe

logger = logging.getLogger(__name__)


def _build_llm() -> LLM:
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return LLM(model=f"openai/{model}", temperature=0.3)


def _build_agents(cfg: dict[str, Any], llm: LLM, retriever_tool: RecipeRetrieverTool) -> dict[str, Agent]:
    agents = {}

    agents["recipe_finder"] = Agent(
        role=cfg["recipe_finder"]["role"],
        goal=cfg["recipe_finder"]["goal"],
        backstory=cfg["recipe_finder"]["backstory"],
        tools=[retriever_tool],
        llm=llm,
        verbose=cfg["recipe_finder"].get("verbose", True),
        allow_delegation=cfg["recipe_finder"].get("allow_delegation", False),
    )

    agents["inventory_agent"] = Agent(
        role=cfg["inventory_agent"]["role"],
        goal=cfg["inventory_agent"]["goal"],
        backstory=cfg["inventory_agent"]["backstory"],
        llm=llm,
        verbose=cfg["inventory_agent"].get("verbose", True),
        allow_delegation=cfg["inventory_agent"].get("allow_delegation", False),
    )

    agents["nutrition_agent"] = Agent(
        role=cfg["nutrition_agent"]["role"],
        goal=cfg["nutrition_agent"]["goal"],
        backstory=cfg["nutrition_agent"]["backstory"],
        llm=llm,
        verbose=cfg["nutrition_agent"].get("verbose", True),
        allow_delegation=cfg["nutrition_agent"].get("allow_delegation", False),
    )

    agents["formatter_agent"] = Agent(
        role=cfg["formatter_agent"]["role"],
        goal=cfg["formatter_agent"]["goal"],
        backstory=cfg["formatter_agent"]["backstory"],
        llm=llm,
        verbose=cfg["formatter_agent"].get("verbose", True),
        allow_delegation=cfg["formatter_agent"].get("allow_delegation", False),
    )

    return agents


def _build_tasks(
    cfg: dict[str, Any],
    agents: dict[str, Agent],
    query: str,
    inventory: list[str],
) -> list[Task]:
    inventory_str = ", ".join(inventory) if inventory else "not specified (assume all ingredients available)"

    retrieve_task = Task(
        description=cfg["retrieve_recipe_task"]["description"].format(query=query),
        expected_output=cfg["retrieve_recipe_task"]["expected_output"],
        agent=agents["recipe_finder"],
    )

    inventory_task = Task(
        description=cfg["inventory_check_task"]["description"].format(inventory=inventory_str),
        expected_output=cfg["inventory_check_task"]["expected_output"],
        agent=agents["inventory_agent"],
        context=[retrieve_task],
    )

    nutrition_task = Task(
        description=cfg["nutrition_task"]["description"],
        expected_output=cfg["nutrition_task"]["expected_output"],
        agent=agents["nutrition_agent"],
        context=[retrieve_task, inventory_task],
    )

    format_task = Task(
        description=cfg["format_output_task"]["description"],
        expected_output=cfg["format_output_task"]["expected_output"],
        agent=agents["formatter_agent"],
        context=[retrieve_task, inventory_task, nutrition_task],
    )

    return [retrieve_task, inventory_task, nutrition_task, format_task]


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _run_crew_with_retry(crew: Crew) -> Any:
    """Run the crew with retry logic for transient LLM failures."""
    return crew.kickoff()


def run_recipe_crew(query: str, inventory: list[str] | None = None) -> dict[str, Any]:
    """
    Orchestrate the full CrewAI pipeline for recipe generation.

    Args:
        query: User's natural language recipe request.
        inventory: Optional list of available ingredients.

    Returns:
        Parsed recipe dict.
    """
    if inventory is None:
        inventory = []

    logger.info(f"Starting recipe crew for query: '{query}' | inventory: {inventory}")

    agents_cfg = load_agents_config()
    tasks_cfg = load_tasks_config()
    llm = _build_llm()
    retriever_tool = RecipeRetrieverTool()

    agents = _build_agents(agents_cfg, llm, retriever_tool)
    tasks = _build_tasks(tasks_cfg, agents, query, inventory)

    crew = Crew(
        agents=list(agents.values()),
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )

    try:
        result = _run_crew_with_retry(crew)
        raw_output = result.raw if hasattr(result, "raw") else str(result)
        logger.info("Crew execution completed successfully.")
    except Exception as e:
        logger.error(f"Crew execution failed after retries: {e}")
        raise

    return safe_parse_recipe(raw_output)
