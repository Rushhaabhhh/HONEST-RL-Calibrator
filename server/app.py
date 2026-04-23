"""FastAPI server entry-point for the HONEST environment."""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from openenv.core.env_server.http_server import create_fastapi_app

from models.models import HonestAction, HonestObservation
from server.environment import DOMAINS, HonestEnvironment

# ---------------------------------------------------------------------------
# Build the base OpenEnv FastAPI app
# create_fastapi_app expects a factory callable, not an instance
# ---------------------------------------------------------------------------

app: FastAPI = create_fastapi_app(
    env=HonestEnvironment,          # factory — called per session
    action_cls=HonestAction,
    observation_cls=HonestObservation,
    max_concurrent_envs=32,
)

# ---------------------------------------------------------------------------
# Custom endpoints
# ---------------------------------------------------------------------------


@app.get("/health", tags=["meta"])
async def health() -> JSONResponse:
    """Docker / load-balancer health check."""
    return JSONResponse({"status": "ok"})


@app.get("/info", tags=["meta"])
async def info() -> JSONResponse:
    """Return project metadata."""
    return JSONResponse(
        {
            "name": "HONEST-Env",
            "version": "0.1.0",
            "description": (
                "Honesty-Optimised and Normalized Environment for Self-Triage — "
                "a calibration benchmark for LLM agents."
            ),
            "domains": DOMAINS,
            "difficulty_range": {"min": 1, "max": 5},
            "episode_length": 5,
            "reward_scheme": "brier_score + format_bonus",
        }
    )
