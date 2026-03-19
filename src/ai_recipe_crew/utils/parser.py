"""
Safe JSON parser with fallback and retry logic for LLM outputs.
"""
import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Expected schema for a recipe response
RECIPE_SCHEMA = {
    "name": "",
    "ingredients": [],
    "steps": [],
    "time": "",
    "nutrition": {
        "calories": "",
        "protein": "",
        "carbs": "",
        "fat": "",
    },
}


def extract_json_from_text(text: str) -> str:
    """
    Attempt to extract a JSON object from text that may contain
    surrounding prose or markdown code blocks.
    """
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    text = re.sub(r"```\s*$", "", text).strip()

    # Try to find the first { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)

    return text


def safe_parse_recipe(raw: str) -> dict[str, Any]:
    """
    Safely parse LLM output into a recipe dict.
    Falls back to a default schema on failure.

    Args:
        raw: Raw string output from the LLM/CrewAI crew.

    Returns:
        Parsed recipe dict conforming to the expected schema.
    """
    if not raw or not raw.strip():
        logger.warning("Received empty output from crew. Returning default schema.")
        return RECIPE_SCHEMA.copy()

    cleaned = extract_json_from_text(raw)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode failed: {e}. Raw output (first 500 chars): {raw[:500]}")
        return _build_fallback(raw)

    return _validate_and_fill(parsed)


def _validate_and_fill(data: dict[str, Any]) -> dict[str, Any]:
    """Ensure all required fields exist, filling defaults for missing ones."""
    result = RECIPE_SCHEMA.copy()
    result["nutrition"] = RECIPE_SCHEMA["nutrition"].copy()

    result["name"] = str(data.get("name", "Unknown Recipe"))
    result["time"] = str(data.get("time", "Unknown"))

    ingredients = data.get("ingredients", [])
    result["ingredients"] = ingredients if isinstance(ingredients, list) else [str(ingredients)]

    steps = data.get("steps", [])
    result["steps"] = steps if isinstance(steps, list) else [str(steps)]

    nutrition = data.get("nutrition", {})
    if isinstance(nutrition, dict):
        result["nutrition"]["calories"] = str(nutrition.get("calories", "N/A"))
        result["nutrition"]["protein"] = str(nutrition.get("protein", "N/A"))
        result["nutrition"]["carbs"] = str(nutrition.get("carbs", "N/A"))
        result["nutrition"]["fat"] = str(nutrition.get("fat", "N/A"))

    return result


def _build_fallback(raw: str) -> dict[str, Any]:
    """Build a minimal fallback response when parsing completely fails."""
    fallback = RECIPE_SCHEMA.copy()
    fallback["nutrition"] = RECIPE_SCHEMA["nutrition"].copy()
    fallback["name"] = "Recipe (parse error)"
    fallback["steps"] = [raw[:500]] if raw else ["Could not parse recipe output."]
    return fallback
