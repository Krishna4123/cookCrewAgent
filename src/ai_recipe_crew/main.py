"""
FastAPI entry point for the AI Recipe Crew microservice.
"""
import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

load_dotenv()

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG pipeline on startup."""
    logger.info("Initializing RAG pipeline...")
    try:
        from ai_recipe_crew.rag.ingest import ingest_recipes
        ingest_recipes()
        logger.info("RAG pipeline ready.")
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        raise
    yield
    logger.info("Shutting down AI Recipe Crew service.")


app = FastAPI(
    title="AI Recipe Crew Service",
    description="CrewAI-powered cooking recipe recommendation microservice with RAG.",
    version="0.1.0",
    lifespan=lifespan,
)


# --- Request / Response Models ---

class RecipeRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Natural language recipe request.")
    inventory: list[str] = Field(default=[], description="Optional list of available ingredients.")


class NutritionInfo(BaseModel):
    calories: str
    protein: str
    carbs: str
    fat: str


class RecipeResponse(BaseModel):
    name: str
    ingredients: list[str]
    steps: list[str]
    time: str
    nutrition: NutritionInfo


# --- Routes ---

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "ai-recipe-crew"}


@app.post("/generate", response_model=RecipeResponse)
async def generate_recipe(request: RecipeRequest):
    """
    Generate a recipe recommendation based on user query and optional inventory.
    """
    logger.info(f"POST /generate | query='{request.query}' | inventory={request.inventory}")

    try:
        from ai_recipe_crew.crew import run_recipe_crew
        result = run_recipe_crew(query=request.query, inventory=request.inventory)
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Recipe generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Recipe generation failed. Please try again.")

    return JSONResponse(content=result)
