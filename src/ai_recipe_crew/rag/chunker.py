"""
Chunking logic for recipe data.
Splits each recipe into separate chunks: name/metadata, ingredients, and steps.
"""
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def chunk_recipe(recipe: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Split a single recipe into multiple chunks for embedding.
    Returns a list of chunk dicts with 'text' and 'metadata'.
    """
    name = recipe.get("name", "Unknown Recipe")
    cuisine = recipe.get("cuisine", "")
    tags = recipe.get("tags", [])
    time = recipe.get("time", "")
    nutrition = recipe.get("nutrition", {})

    base_metadata = {
        "recipe_name": name,
        "cuisine": cuisine,
        "tags": json.dumps(tags),
        "time": time,
    }

    chunks = []

    # Chunk 1: Overview (name + cuisine + tags + time)
    overview_text = (
        f"Recipe: {name}. "
        f"Cuisine: {cuisine}. "
        f"Tags: {', '.join(tags)}. "
        f"Cooking time: {time}."
    )
    chunks.append({
        "text": overview_text,
        "metadata": {**base_metadata, "chunk_type": "overview"},
    })

    # Chunk 2: Ingredients
    ingredients = recipe.get("ingredients", [])
    ingredients_text = (
        f"Recipe: {name}. "
        f"Ingredients needed: {', '.join(ingredients)}."
    )
    chunks.append({
        "text": ingredients_text,
        "metadata": {
            **base_metadata,
            "chunk_type": "ingredients",
            "ingredients": json.dumps(ingredients),
        },
    })

    # Chunk 3: Steps
    steps = recipe.get("steps", [])
    steps_text = (
        f"Recipe: {name}. "
        f"Cooking steps: {' '.join(f'Step {i+1}: {s}' for i, s in enumerate(steps))}"
    )
    chunks.append({
        "text": steps_text,
        "metadata": {
            **base_metadata,
            "chunk_type": "steps",
            "steps": json.dumps(steps),
        },
    })

    # Chunk 4: Nutrition
    nutrition_text = (
        f"Recipe: {name}. "
        f"Nutrition per serving — "
        f"Calories: {nutrition.get('calories', 'N/A')}, "
        f"Protein: {nutrition.get('protein', 'N/A')}, "
        f"Carbs: {nutrition.get('carbs', 'N/A')}, "
        f"Fat: {nutrition.get('fat', 'N/A')}."
    )
    chunks.append({
        "text": nutrition_text,
        "metadata": {
            **base_metadata,
            "chunk_type": "nutrition",
            "calories": nutrition.get("calories", ""),
            "protein": nutrition.get("protein", ""),
            "carbs": nutrition.get("carbs", ""),
            "fat": nutrition.get("fat", ""),
        },
    })

    return chunks


def chunk_all_recipes(recipes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chunk all recipes and return a flat list of all chunks."""
    all_chunks = []
    for recipe in recipes:
        try:
            chunks = chunk_recipe(recipe)
            all_chunks.extend(chunks)
        except Exception as e:
            logger.warning(f"Failed to chunk recipe '{recipe.get('name')}': {e}")
    logger.info(f"Generated {len(all_chunks)} chunks from {len(recipes)} recipes.")
    return all_chunks
