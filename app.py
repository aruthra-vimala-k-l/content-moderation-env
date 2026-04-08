"""
FastAPI application for the Content Moderation OpenEnv Environment.

The app is created via openenv.core's create_fastapi_app helper, which
automatically registers the following HTTP endpoints:

  POST /reset          – start a new episode, returns ModerationObservation
  POST /step           – submit an action, returns ModerationObservation
  GET  /state          – return current ModerationState
  GET  /health         – health check
  GET  /schema/action  – JSON schema for ModerationAction
  GET  /schema/observation – JSON schema for ModerationObservation
  GET  /docs           – Swagger UI
  GET  /redoc          – ReDoc UI

Reset accepts an optional JSON body:
  { "task": "binary-classification" | "multi-label-toxicity" | "contextual-moderation",
    "seed": <int | null> }
"""

import uvicorn
from openenv.core.env_server import create_fastapi_app

from environment import ContentModerationEnvironment
from models import ModerationAction, ModerationObservation

# create_fastapi_app expects a factory callable (not an instance)
app = create_fastapi_app(
    env=ContentModerationEnvironment,
    action_cls=ModerationAction,
    observation_cls=ModerationObservation,
    max_concurrent_envs=10,
)

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
        log_level="info",
    )
